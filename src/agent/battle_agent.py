"""
Battle Agent — Combat heuristique Gen 1 (sans ML).

Lit la RAM directement pour naviguer les menus de combat.

Stratégie (A-spam avec sélection de move) :
  0xD11C n'est pas fiable en combat — on utilise une file d'actions à la place.

  Séquence par tour :
    1. A × _INTRO_PRESSES   → passe les dialogs d'intro / "What will X do?"
    2. A                    → sélectionne FIGHT (curseur toujours sur FIGHT)
    3. DOWN × idx           → navigue vers le meilleur move (liste verticale)
    4. A                    → exécute le move
    5. A × _POST_PRESSES    → passe animations + tour ennemi

Gen 1 move menu layout (liste verticale, 1 colonne) :
    Move 0  ← curseur ici
    Move 1
    Move 2
    Move 3
"""

from __future__ import annotations
from pyboy import PyBoy
from src.emulator.ram_map import (
    RAM_ENEMY_TYPE1, RAM_ENEMY_TYPE2,
    RAM_MOVE_IDS, RAM_MOVE_PP,
)

# ─── Gen 1 type constants ─────────────────────────────────────────────────────
T_NORMAL   = 0x00
T_FIGHT    = 0x01
T_FLYING   = 0x02
T_POISON   = 0x03
T_GROUND   = 0x04
T_ROCK     = 0x05
T_BUG      = 0x07
T_GHOST    = 0x08
T_FIRE     = 0x14
T_WATER    = 0x15
T_GRASS    = 0x16
T_ELECTRIC = 0x17
T_PSYCHIC  = 0x18
T_ICE      = 0x19
T_DRAGON   = 0x1A

TYPE_CHART: dict[int, set[int]] = {
    T_WATER:    {T_FIRE, T_ROCK, T_GROUND},
    T_GRASS:    {T_WATER, T_ROCK, T_GROUND},
    T_FIRE:     {T_GRASS, T_BUG, T_ICE},
    T_FIGHT:    {T_NORMAL, T_ROCK, T_ICE},
    T_GROUND:   {T_FIRE, T_ELECTRIC, T_POISON, T_ROCK},
    T_ROCK:     {T_FIRE, T_FLYING, T_BUG, T_ICE},
    T_ELECTRIC: {T_WATER, T_FLYING},
    T_PSYCHIC:  {T_FIGHT, T_POISON},
    T_ICE:      {T_GRASS, T_GROUND, T_FLYING, T_DRAGON},
    T_BUG:      {T_GRASS, T_POISON, T_PSYCHIC},
}

MOVE_TYPES: dict[int, int] = {
    0x01: T_NORMAL,    # Pound
    0x02: T_FIGHT,     # Karate Chop
    0x05: T_NORMAL,    # Mega Punch
    0x0A: T_NORMAL,    # Pay Day
    0x0C: T_FIGHT,     # Double Kick
    0x1C: T_FIRE,      # Ember
    0x1E: T_WATER,     # Surf
    0x21: T_GRASS,     # Vine Whip
    0x22: T_GRASS,     # Absorb
    0x23: T_GRASS,     # Mega Drain
    0x27: T_WATER,     # Water Gun
    0x28: T_WATER,     # Bubble Beam
    0x2D: T_NORMAL,    # Quick Attack
    0x33: T_NORMAL,    # Scratch
    0x3D: T_GRASS,     # Razor Leaf
    0x49: T_WATER,     # Bubble
    0x4D: T_NORMAL,    # Tackle
    0x5B: T_FIGHT,     # Low Kick
    0x62: T_NORMAL,    # Growl  (status)
}

STATUS_MOVES = {0x62, 0x21, 0x45, 0x4E, 0x50, 0x6F}   # Growl, Smokescreen, etc.


class BattleAgent:
    """Heuristic battle agent for Gen 1 Pokémon."""

    _INTRO_PRESSES = 8    # A à spammer pour les dialogs d'intro
    _POST_PRESSES  = 12   # A après le move (animations + tour ennemi)

    def __init__(self):
        self._queue: list[str] = []
        self._turn: int        = 0

    def reset(self):
        self._queue = ['a'] * self._INTRO_PRESSES
        self._turn  = 0

    def act(self, pyboy: PyBoy) -> str | None:
        if self._queue:
            return self._queue.pop(0)
        self._build_turn_queue(pyboy)
        return self._queue.pop(0) if self._queue else None

    def _build_turn_queue(self, pyboy: PyBoy):
        idx = self._best_move_index(pyboy)
        self._turn += 1
        self._queue = (
            ['a']               # FIGHT
            + ['down'] * idx    # navigate to move
            + ['a']             # execute
            + ['a'] * self._POST_PRESSES
        )

    def _best_move_index(self, pyboy: PyBoy) -> int:
        enemy_types = {pyboy.memory[RAM_ENEMY_TYPE1], pyboy.memory[RAM_ENEMY_TYPE2]}
        best_idx, best_score = 0, -1

        for i in range(4):
            pp      = pyboy.memory[RAM_MOVE_PP[i]]
            move_id = pyboy.memory[RAM_MOVE_IDS[i]]
            if pp == 0 or move_id == 0:
                continue
            if move_id in STATUS_MOVES:
                score = 0
            else:
                move_type = MOVE_TYPES.get(move_id, T_NORMAL)
                score = 2 if TYPE_CHART.get(move_type, set()) & enemy_types else 1

            if score > best_score:
                best_score, best_idx = score, i

        return best_idx
