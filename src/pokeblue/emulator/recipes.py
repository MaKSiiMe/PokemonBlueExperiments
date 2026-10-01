"""Savestates de référence régénérables : savestate de base + recette d'inputs.

Un manifeste YAML (ex. `configs/states/modes.yaml`) liste des `StateRecipe`. Les
savestates de base (`states/`) et la ROM ne sont jamais versionnés ; les recettes le
sont, ce qui rend chaque savestate dérivé reproductible à l'identique.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from pokeblue.emulator.core import Emulator, parse_recipe


@dataclass(frozen=True, slots=True)
class StateRecipe:
    name: str
    base: str            # fichier de savestate de base, relatif au dossier des states
    recipe: str          # jetons séparés par des espaces (voir parse_recipe)
    mode: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        parse_recipe(self.recipe)  # valide la recette dès le chargement


def load_manifest(path: str | Path) -> list[StateRecipe]:
    """Charge un manifeste ; une recette peut être une chaîne ou une liste de chaînes."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    entries = []
    for raw in data["states"]:
        recipe = raw["recipe"]
        if isinstance(recipe, list):
            recipe = " ".join(recipe)
        entries.append(StateRecipe(
            name=raw["name"], base=raw["base"], recipe=recipe,
            mode=raw.get("mode"), note=raw.get("note", ""),
        ))
    names = [e.name for e in entries]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"{path} : noms en double {sorted(duplicates)}")
    return entries


def play(emu: Emulator, entry: StateRecipe, states_dir: str | Path) -> None:
    """Charge le savestate de base de `entry` et rejoue sa recette."""
    emu.load_state(Path(states_dir) / entry.base)
    emu.run(entry.recipe)
