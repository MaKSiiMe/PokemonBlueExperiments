"""Graphe de progression (progression.yaml) : jalons, état d'avancement, jalon suivant."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import yaml

from pokeblue.knowledge.navigation import Ability, Progress
from pokeblue.knowledge.trainers import (
    TrainerMon,
    max_level,
    rival_party,
    starter_name,
    trainer_team,
)

PROGRESSION_FILE = Path(__file__).parent / "progression.yaml"


@dataclass(frozen=True)
class Milestone:
    id: str
    description: str
    target_map: str
    trainer_class: str | None = None
    trainer_party: int | None = None
    rival_base: int | None = None
    completes_when: tuple[dict, ...] = ()        # une seule condition suffit
    requires_milestones: tuple[str, ...] = ()
    requires_abilities: Ability = Ability.NONE
    requires: dict = field(default_factory=dict)  # objets / badges exigés
    recommended_level: int | None = None
    optional: bool = False

    def done(self, progress: Progress) -> bool:
        return any(progress.satisfies(cond) for cond in self.completes_when)

    def trainer_team(self, rival_starter: str | None = None) -> tuple[TrainerMon, ...] | None:
        if self.trainer_class is None:
            return None
        if self.rival_base is not None:
            starter = rival_starter or "SQUIRTLE"
            return trainer_team(self.trainer_class, rival_party(self.rival_base, starter), starter)
        return trainer_team(self.trainer_class, self.trainer_party)


@cache
def milestones() -> tuple[Milestone, ...]:
    raw = yaml.safe_load(PROGRESSION_FILE.read_text(encoding="utf-8"))["milestones"]
    out = []
    for m in raw:
        target = m["target"]
        trainer = target.get("trainer") or {}
        requires = dict(m.get("requires", {}))
        abilities = Ability.NONE
        for name in requires.pop("abilities", []):
            abilities |= Ability[name]
        conditions = m.get("completes_when_any") or [m["completes_when"]]
        out.append(Milestone(
            id=m["id"], description=m["description"], target_map=target["map"],
            trainer_class=trainer.get("class"), trainer_party=trainer.get("party"),
            rival_base=trainer.get("rival_base"), completes_when=tuple(conditions),
            requires_milestones=tuple(requires.pop("milestones", [])),
            requires_abilities=abilities, requires=requires,
            recommended_level=m.get("recommended_level"), optional=m.get("optional", False),
        ))
    return tuple(out)


@cache
def milestone(milestone_id: str) -> Milestone:
    return next(m for m in milestones() if m.id == milestone_id)


def required_level(m: Milestone | str, rival_starter: str | None = None) -> int:
    """Niveau conseillé : explicite, sinon niveau maximal du dresseur ciblé, sinon celui
    du jalon précédent dans l'ordre du fichier."""
    m = milestone(m) if isinstance(m, str) else m
    if m.recommended_level is not None:
        return m.recommended_level
    team = m.trainer_team(rival_starter)
    if team:
        return max_level(team)
    index = milestones().index(m)
    return required_level(milestones()[index - 1], rival_starter) if index else 5


def completed(progress: Progress) -> tuple[Milestone, ...]:
    return tuple(m for m in milestones() if m.done(progress))


def next_milestone(state_or_progress, include_optional: bool = False) -> Milestone | None:
    """Premier jalon non accompli (dans l'ordre du fichier) dont les jalons préalables
    sont accomplis ; None une fois le Champion battu."""
    progress = (state_or_progress if isinstance(state_or_progress, Progress)
                else Progress.from_state(state_or_progress))
    done = {m.id for m in completed(progress)}
    for m in milestones():
        if m.id in done or (m.optional and not include_optional):
            continue
        if all(req in done for req in m.requires_milestones):
            return m
    return None


def missing_requirements(m: Milestone, progress: Progress) -> list[str]:
    """Ce qui manque pour tenter le jalon (capacités, objets, badges, jalons)."""
    done = {x.id for x in completed(progress)}
    missing = [f"jalon {r}" for r in m.requires_milestones if r not in done]
    missing += [f"capacité {a.name}" for a in Ability if a and a in m.requires_abilities
                and not progress.abilities & a]
    if m.requires and not progress.satisfies(m.requires):
        missing.append(f"conditions {m.requires}")
    return missing


def rival_starter_name(state) -> str | None:
    return starter_name(state.rival_starter) if state.rival_starter else None
