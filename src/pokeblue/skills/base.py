"""Contrat commun des skills (scriptés ou appris).

Un skill reçoit un objectif (`Goal`), puis, à chaque pas, le `GameState` courant ; il
renvoie un bouton à presser (une action de 24 frames, voir pokeblue.emulator) ou
`None` pour laisser le jeu avancer. Il rend compte de son état via `status()`. Un
skill qui dépasse `budget_steps` passe en TIMEOUT : l'orchestrateur reprend la main.
Passer d'une version scriptée à une version apprise ne change pas l'orchestrateur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol

from pokeblue.state.game_state import GameState

Button = str   # "up", "down", "left", "right", "a", "b", "start", "select"


class SkillStatus(Enum):
    RUNNING = auto()
    SUCCESS = auto()
    FAILURE = auto()
    TIMEOUT = auto()


@dataclass(frozen=True)
class Goal:
    kind: str
    params: dict = field(default_factory=dict)

    def __getitem__(self, key: str):
        return self.params[key]

    def get(self, key: str, default=None):
        return self.params.get(key, default)


class Skill(Protocol):
    name: str
    budget_steps: int

    def start(self, goal: Goal, state: GameState) -> None: ...
    def act(self, state: GameState) -> Button | None: ...
    def status(self, state: GameState) -> SkillStatus: ...


class BaseSkill:
    """Socle commun : objectif courant, compteur de pas, TIMEOUT, motif d'échec."""

    name = "skill"
    budget_steps = 1000

    def __init__(self) -> None:
        self.goal: Goal | None = None
        self.steps = 0
        self.failure: str | None = None
        self.done = False

    def start(self, goal: Goal, state: GameState) -> None:
        self.goal, self.steps, self.failure, self.done = goal, 0, None, False

    def fail(self, reason: str) -> None:
        self.failure = reason

    def act(self, state: GameState) -> Button | None:
        self.steps += 1
        return self._act(state)

    def _act(self, state: GameState) -> Button | None:
        raise NotImplementedError

    def status(self, state: GameState) -> SkillStatus:
        if self.failure:
            return SkillStatus.FAILURE
        if self.done or self._succeeded(state):
            return SkillStatus.SUCCESS
        if self.steps >= self.budget_steps:
            return SkillStatus.TIMEOUT
        return SkillStatus.RUNNING

    def _succeeded(self, state: GameState) -> bool:
        return False
