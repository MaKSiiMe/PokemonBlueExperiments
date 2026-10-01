"""Émulateur PyBoy avec les conventions du projet.

Une *action* dure `TICKS_PER_ACTION` frames : le bouton est maintenu un temps court
(`HOLD_FRAMES` pour une direction, `PRESS_FRAMES` sinon) puis relâché, et le reste de
l'action laisse le jeu avancer. Une recette d'inputs est donc rejouable telle quelle
par l'environnement RL, qui utilise les mêmes actions.

Durées mesurées sur savestates (Pewter City, 24 décalages de phase) :
  - le jeu ne lit la manette qu'une frame sur deux : un appui d'une frame est perdu
    une fois sur deux ; à partir de 2 frames il ne l'est jamais ;
  - une direction maintenue de 2 à 17 frames fait exactement un pas (16 frames),
    au-delà le jeu enchaîne un second pas (23 frames = 2 cases).

Recettes : suite de jetons séparés par des espaces (ou liste de jetons) —
    `a`, `b`, `start`, `select`, `up`, `down`, `left`, `right`  une action
    `up*3`                                                      action répétée
    `wait:60`                                                   60 frames sans input
    `hold:left:6`                                               bouton maintenu 6 frames,
                                                                sans compléter l'action
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from pathlib import Path

from pyboy import PyBoy

from pokeblue.state.memory import HIGH_END, HIGH_START, WRAM_END, WRAM_START, MemorySnapshot

ACTIONS = ("up", "down", "left", "right", "a", "b", "start", "select")
DIRECTIONS = frozenset({"up", "down", "left", "right"})
TICKS_PER_ACTION = 24   # ~0,4 s : un pas complet (16 frames) et sa fin
HOLD_FRAMES = 8         # appui d'une direction : exactement un pas
PRESS_FRAMES = 4        # appui des autres boutons


def parse_recipe(recipe: str | Iterable[str]) -> list[tuple[str, str, int]]:
    """Analyse une recette d'inputs en étapes (genre, bouton, frames).

    Genres : ("act", bouton, 1), ("wait", "", frames), ("hold", bouton, frames).

    Raises:
        ValueError: jeton inconnu.
    """
    tokens = recipe.split() if isinstance(recipe, str) else list(recipe)
    steps: list[tuple[str, str, int]] = []
    for token in tokens:
        kind, _, rest = token.partition(":")
        if kind == "wait" and rest.isdigit():
            steps.append(("wait", "", int(rest)))
            continue
        if kind == "hold":
            button, _, frames = rest.partition(":")
            if button in ACTIONS and frames.isdigit():
                steps.append(("hold", button, int(frames)))
                continue
        name, _, count = token.partition("*")
        if name not in ACTIONS or (count and not count.isdigit()):
            raise ValueError(f"jeton de recette inconnu : {token!r}")
        steps += [("act", name, 1)] * (int(count) if count else 1)
    return steps


class Emulator:
    """Instance PyBoy headless (par défaut) pilotée par actions et recettes."""

    def __init__(self, rom_path: str | Path, *, window: str = "null", speed: int = 0) -> None:
        self.pyboy = PyBoy(str(rom_path), window=window, sound=False)
        self.pyboy.set_emulation_speed(speed)

    # ── Savestates ────────────────────────────────────────────────────────────

    def load_state(self, state: str | Path | bytes) -> None:
        if isinstance(state, bytes):
            self.pyboy.load_state(io.BytesIO(state))
        else:
            with open(state, "rb") as f:
                self.pyboy.load_state(f)

    def save_state(self) -> bytes:
        buf = io.BytesIO()
        self.pyboy.save_state(buf)
        return buf.getvalue()

    def save_state_to(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(self.save_state())

    # ── Inputs ────────────────────────────────────────────────────────────────

    def tick(self, frames: int = 1, render: bool = False) -> None:
        if frames > 0:
            self.pyboy.tick(frames, render)

    def act(self, button: str, render: bool = False) -> None:
        """Une action de `TICKS_PER_ACTION` frames (voir le docstring du module)."""
        if button not in ACTIONS:
            raise ValueError(f"bouton inconnu : {button!r}")
        hold = HOLD_FRAMES if button in DIRECTIONS else PRESS_FRAMES
        self.pyboy.button(button, delay=hold)
        self.tick(TICKS_PER_ACTION - 1)
        self.tick(1, render)

    def run(self, recipe: str | Iterable[str], render: bool = False) -> None:
        """Rejoue une recette d'inputs (voir `parse_recipe`)."""
        for kind, button, frames in parse_recipe(recipe):
            if kind == "wait":
                self.tick(frames, render)
            elif kind == "hold":
                self.pyboy.button_press(button)
                self.tick(frames, render)
                self.pyboy.button_release(button)
            else:
                self.act(button, render)

    # ── Lecture ───────────────────────────────────────────────────────────────

    def snapshot(self) -> MemorySnapshot:
        """WRAM et page haute lues en une passe."""
        memory = self.pyboy.memory
        return MemorySnapshot(
            bytes(memory[WRAM_START:WRAM_END + 1]),
            bytes(memory[HIGH_START:HIGH_END + 1]),
        )

    def screen_image(self):
        """Image PIL RGB de l'écran (160×144).

        PyBoy ne met l'image à jour que sur les frames rendues : on avance donc d'une
        frame avec rendu avant la capture.
        """
        self.tick(1, render=True)
        return self.pyboy.screen.image.convert("RGB")

    def close(self) -> None:
        self.pyboy.stop(save=False)

    def __enter__(self) -> Emulator:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
