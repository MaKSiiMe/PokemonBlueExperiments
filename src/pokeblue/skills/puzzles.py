"""Énigmes scriptées : les poubelles de l'arène de Carmin.

Résolution « de joueur informé » : on fouille les poubelles jusqu'au 1er interrupteur
(EVENT_1ST_LOCK_OPENED), puis on ne teste que ses voisines possibles (table
GymTrashCans de pokered, voir puzzles.yaml). Un échec remet les interrupteurs à zéro
et place le 1er dans une poubelle d'indice pair. Les index tirés au sort
(wFirstLockTrashCanIndex…) ne sont jamais lus : l'agent n'a que ce qu'un joueur voit.
"""

from __future__ import annotations

from pokeblue.knowledge.gen1_data import EVENT_IDS
from pokeblue.knowledge.navigation import Navigator
from pokeblue.knowledge.puzzles import puzzle
from pokeblue.skills.base import BaseSkill, Button, Goal, SkillStatus
from pokeblue.skills.dialog import dialog_button
from pokeblue.skills.navigation import Navigating, NavigationSkill, facing_squares
from pokeblue.state.game_state import GameState
from pokeblue.state.mode_detector import Mode, detect_mode

FIRST_LOCK = EVENT_IDS["EVENT_1ST_LOCK_OPENED"]
SECOND_LOCK = EVENT_IDS["EVENT_2ND_LOCK_OPENED"]


class TrashCanSkill(Navigating, BaseSkill):
    name = "trash_cans"
    budget_steps = 3000
    handles_menus = True

    def start(self, goal: Goal, state: GameState) -> None:
        super().start(goal, state)
        self.data = puzzle("vermilion_trash_cans")
        self.map = self.data["map"]
        self.nav = NavigationSkill()
        self.pending: int | None = None      # poubelle en cours de fouille
        self.first: int | None = None        # poubelle du 1er interrupteur
        self.tried: set[int] = set()
        self.first_candidates = list(range(len(self.data["cans"])))

    def _next_can(self) -> int | None:
        if self.first is None:
            options = [c for c in self.first_candidates if c not in self.tried]
        else:
            neighbors = self.data["second_lock_candidates"][self.first]
            options = [c for c in [*neighbors, self.data["second_lock_bug_can"]]
                       if c not in self.tried and c != self.first]
        return options[0] if options else None

    def _resolve(self, state: GameState) -> None:
        """Résultat de la dernière fouille, lu dans les drapeaux d'événement."""
        can, self.pending = self.pending, None
        if self.first is None:
            self.tried.add(can)
            if state.flag(FIRST_LOCK):
                self.first, self.tried = can, set()
        elif not state.flag(FIRST_LOCK):          # mauvaise poubelle : tout est remis à zéro
            self.first, self.tried = None, set()
            if self.data["first_lock_after_reset"] == "even":
                self.first_candidates = list(range(0, len(self.data["cans"]), 2))
        else:
            self.tried.add(can)

    def _act(self, state: GameState) -> Button | None:
        mode = detect_mode(state)
        if mode is not Mode.OVERWORLD:
            return dialog_button(state)
        if state.flag(SECOND_LOCK):
            self.done = True
            return None
        if self.pending is not None and self.nav.status(state) is SkillStatus.SUCCESS:
            self._resolve(state)
        if self.pending is None:
            can = self._next_can()
            if can is None:
                self.first, self.tried = None, set()      # recommencer la fouille
                can = self._next_can()
            self.pending = can
            x, y = self.data["cans"][can]
            nav = Navigator.from_state(state)
            targets = facing_squares(nav, self.map, x, y)
            self.nav.start(Goal("navigate", {"map": self.map, "interact": True, "targets": [
                [sq.x, sq.y, face] for sq, face in targets.items()]}), state)
        button = self.nav.act(state)
        if self.nav.status(state) in (SkillStatus.FAILURE, SkillStatus.TIMEOUT):
            self.fail(f"poubelle {self.pending} : {self.nav.failure or 'timeout'}")
        return button

    def _succeeded(self, state: GameState) -> bool:
        return state.flag(SECOND_LOCK) and detect_mode(state) is Mode.OVERWORLD

