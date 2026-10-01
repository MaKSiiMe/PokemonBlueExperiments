"""Orchestrateur : boucle objectif → skill → boutons, sans jamais rester bloqué.

À chaque pas :
1. combat en cours → BattleSkill (objectif fourni par la stratégie) ;
2. transition (fondu, écran noir) → on laisse le jeu avancer ;
3. point de décision (pas de skill actif, retour dans l'overworld après une scène,
   un dialogue ou un combat, ou toutes les `REPLAN_EVERY` actions) → la stratégie
   choisit l'objectif ; s'il a changé, le skill correspondant le remplace ;
4. capacité de terrain demandée par la navigation (arbre à couper, eau) →
   FieldMoveSkill, puis reprise de la navigation ;
5. dialogue ou menu que le skill actif ne gère pas → politique de dialogue ;
6. sinon → skill actif.

Tout FAILURE ou TIMEOUT est journalisé (`SkillEvent`) et transmis à `on_failure`
(le runner y sauvegarde un savestate). Après `MAX_SAME_FAILURES` échecs consécutifs
du même objectif, l'orchestrateur intercale une marche aléatoire (WanderSkill) pour
débloquer la situation ; `stuck` signale au runner d'abandonner.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from pokeblue.knowledge.gen1_data import MAPS
from pokeblue.orchestrator.strategy import Strategy
from pokeblue.skills.base import BaseSkill, Button, Goal, SkillStatus
from pokeblue.skills.battle import BattleSkill
from pokeblue.skills.dialog import DialogSkill, dialog_button
from pokeblue.skills.field import HealSkill, TrainSkill
from pokeblue.skills.menus import BuySkill, FieldMoveSkill, TeachMoveSkill
from pokeblue.skills.navigation import NavigationSkill
from pokeblue.skills.puzzles import TrashCanSkill
from pokeblue.state.game_state import GameState
from pokeblue.state.mode_detector import Mode, detect_mode

REPLAN_EVERY = 200
MAX_SAME_FAILURES = 3      # échecs consécutifs d'un objectif avant une marche aléatoire
STUCK_FAILURES = 12        # échecs consécutifs (tous objectifs) : run abandonné


@dataclass(frozen=True)
class SkillEvent:
    step: int
    skill: str
    status: str
    reason: str
    goal: str
    map: str
    x: int
    y: int
    milestone: str | None = None


class WanderSkill(BaseSkill):
    """Repli : quelques pas au hasard (débloque un PNJ en travers, un script en attente)."""

    name = "wander"
    budget_steps = 24

    def __init__(self, rng: random.Random) -> None:
        super().__init__()
        self.rng = rng

    def _act(self, state: GameState) -> Button | None:
        if detect_mode(state) is not Mode.OVERWORLD:
            return dialog_button(state)
        return self.rng.choice(("up", "down", "left", "right"))

    def status(self, state: GameState) -> SkillStatus:
        return SkillStatus.SUCCESS if self.steps >= self.budget_steps else SkillStatus.RUNNING


def goal_key(goal: Goal) -> str:
    params = {k: v for k, v in goal.params.items() if k != "milestone"}
    return f"{goal.kind}{sorted(params.items())}"


@dataclass
class Orchestrator:
    strategy: Strategy = field(default_factory=Strategy)
    on_failure: Callable[[SkillEvent], None] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.skills: dict[str, BaseSkill] = {
            "navigate": NavigationSkill(), "heal": HealSkill(), "train": TrainSkill(),
            "dialog": DialogSkill(), "wander": WanderSkill(self.rng),
            "teach": TeachMoveSkill(), "trash_cans": TrashCanSkill(), "buy": BuySkill(),
        }
        self.field_move = FieldMoveSkill()
        self.sub: BaseSkill | None = None     # capacité de terrain demandée par la navigation
        self.battle = BattleSkill()
        self.in_battle = False
        self.active: BaseSkill | None = None
        self.active_goal: Goal | None = None
        self.decision_point = True
        self.last_decision = 0
        self.steps = 0
        self.events: list[SkillEvent] = []
        self.same_failures: dict[str, int] = {}
        self.consecutive_failures = 0
        self.finished = False
        self.last_success: str | None = None

    @property
    def stuck(self) -> bool:
        return self.consecutive_failures >= STUCK_FAILURES

    # ── Boucle ────────────────────────────────────────────────────────────────

    def step(self, state: GameState) -> Button | None:
        self.steps += 1
        mode = detect_mode(state)

        if state.battle is not None:
            return self._battle_step(state)
        if self.in_battle:
            self.in_battle = False
            self.decision_point = True

        if mode is Mode.TRANSITION:
            return None
        if self.sub is not None:
            return self._sub_step(state)
        if mode is Mode.OVERWORLD and not state.scripted and (
                self.active is None or self.decision_point
                or self.steps - self.last_decision >= REPLAN_EVERY):
            self._decide(state)
            if self.finished:
                return None
        if self.active is None:
            return dialog_button(state)

        if mode in (Mode.DIALOG, Mode.MENU) and not getattr(self.active, "handles_menus", False):
            self.decision_point = True
            return dialog_button(state)
        if state.scripted:
            self.decision_point = True

        button = self.active.act(state)
        status = self.active.status(state)
        if status is not SkillStatus.RUNNING:
            self._finish(state, self.active, status)
        elif getattr(self.active, "needs_field_move", None):
            self.sub = self.field_move
            self.sub.start(Goal("field_move", {"move": self.active.needs_field_move}), state)
        return button

    def _sub_step(self, state: GameState) -> Button | None:
        button = self.sub.act(state)
        status = self.sub.status(state)
        if status is not SkillStatus.RUNNING:
            if status is not SkillStatus.SUCCESS:
                self._log(state, self.sub, status, self.sub.goal)
            self.sub = None
            if self.active is not None:
                self.active.field_move_done()
        return button

    def _battle_step(self, state: GameState) -> Button | None:
        if not self.in_battle:
            self.in_battle = True
            self.battle.start(self.strategy.battle_goal(state), state)
        button = self.battle.act(state)
        status = self.battle.status(state)
        if status in (SkillStatus.FAILURE, SkillStatus.TIMEOUT):
            self._log(state, self.battle, status, self.battle.goal)
            self.battle.start(self.strategy.battle_goal(state), state)
        return button

    # ── Décisions ─────────────────────────────────────────────────────────────

    def _decide(self, state: GameState) -> None:
        self.decision_point = False
        self.last_decision = self.steps
        goal = self.strategy.decide(state)
        if goal.kind == "done":
            self.finished = True
            return
        key = goal_key(goal)
        if self.same_failures.get(key, 0) >= MAX_SAME_FAILURES and self.active_goal != Goal("wander"):
            self.same_failures[key] = 0
            goal = Goal("wander")
        if self.active is not None and self.active_goal == goal:
            return
        self.active_goal = goal
        self.active = self.skills[goal.kind]
        self.active.start(goal, state)

    def _finish(self, state: GameState, skill: BaseSkill, status: SkillStatus) -> None:
        goal = self.active_goal
        key = goal_key(goal)
        if status is SkillStatus.SUCCESS:
            if skill.steps <= 1 and self.last_success == key:
                # Objectif déjà atteint mais jalon inchangé : pas de progrès.
                self._log(state, skill, SkillStatus.FAILURE, goal, "objectif atteint sans progrès")
            else:
                self.same_failures.pop(key, None)
                if goal.kind != "wander":
                    self.consecutive_failures = 0
            self.last_success = key
        else:
            self._log(state, skill, status, goal)
        self.active = None
        self.active_goal = None
        self.decision_point = True

    def _log(self, state: GameState, skill: BaseSkill, status: SkillStatus, goal: Goal | None,
             reason: str | None = None) -> None:
        key = goal_key(goal) if goal else skill.name
        self.same_failures[key] = self.same_failures.get(key, 0) + 1
        self.consecutive_failures += 1
        event = SkillEvent(
            step=self.steps, skill=skill.name, status=status.name,
            reason=reason or skill.failure or f"budget de {skill.budget_steps} actions dépassé",
            goal=key, map=MAPS[state.map_id].name, x=state.x, y=state.y,
            milestone=goal.get("milestone") if goal else None,
        )
        self.events.append(event)
        if self.on_failure:
            self.on_failure(event)
