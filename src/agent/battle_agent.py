"""
Battle Agent — Heuristic combat decision-making.

Does not use ML — reads RAM directly and applies type-advantage logic.

Strategy:
  1. If HP < 30% and potion available → use potion
  2. Otherwise → select the most effective move (highest type advantage, PP > 0)
  3. Navigate Gen 1 battle menus via button sequence queue

RAM addresses:
  0xD057    — Battle type (1=wild, 2=trainer)
  0xD016    — Enemy Pokémon ID
  0xD018    — Enemy level
  0xD01F    — Enemy type 1
  0xD020    — Enemy type 2
  0xCFE6-7  — Enemy current HP (big-endian uint16)
  0xCFE8-9  — Enemy max HP    (big-endian uint16)
  0xD16C-D  — Player current HP (big-endian uint16)
  0xD18C-D  — Player max HP     (big-endian uint16)
  0xD173-6  — Player move IDs (4 bytes)
  0xD188-B  — Player move PP  (4 bytes)
"""

from __future__ import annotations
from pyboy import PyBoy

# ─── RAM addresses ────────────────────────────────────────────────────────────
RAM_ENEMY_ID     = 0xD016
RAM_ENEMY_LVL    = 0xD018
RAM_ENEMY_TYPE1  = 0xD01F
RAM_ENEMY_TYPE2  = 0xD020
RAM_ENEMY_HP_H   = 0xCFE6
RAM_ENEMY_MHP_H  = 0xCFE8
RAM_PLAYER_HP_H  = 0xD16C
RAM_PLAYER_MHP_H = 0xD18C
RAM_MOVE_IDS     = (0xD173, 0xD174, 0xD175, 0xD176)
RAM_MOVE_PP      = (0xD188, 0xD189, 0xD18A, 0xD18B)

# ─── Gen 1 type constants ────────────────────────────────────────────────────
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

# attack_type → set of defending types that take super-effective damage
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

# Partial move-ID → type table (Gen 1 early-game moves)
MOVE_TYPES: dict[int, int] = {
    0x01: T_NORMAL,    # Pound
    0x02: T_FIGHT,     # Karate Chop
    0x0C: T_FIGHT,     # Double Kick
    0x1C: T_FIRE,      # Ember
    0x1E: T_WATER,     # Surf
    0x21: T_GRASS,     # Vine Whip
    0x27: T_WATER,     # Water Gun
    0x33: T_NORMAL,    # Scratch
    0x49: T_WATER,     # Bubble
    0x4D: T_NORMAL,    # Tackle
    0x5B: T_FIGHT,     # Low Kick
    0x3D: T_GRASS,     # Razor Leaf
    0x62: T_NORMAL,    # Growl (status — low priority)
    0x2D: T_NORMAL,    # Quick Attack
}

HP_POTION_THRESHOLD = 0.30   # Use potion if HP < 30%


class BattleAgent:
    """
    Heuristic-based battle agent. No training required.
    Reads RAM to select moves and manage items.
    """

    def __init__(self):
        self._queue: list[str] = []   # Pending button presses

    # ── Public API ────────────────────────────────────────────────────────────

    def act(self, pyboy: PyBoy) -> str | None:
        """
        Returns the next button to press, or None to pass this tick.
        Manages its own internal menu-navigation queue.
        """
        if self._queue:
            return self._queue.pop(0)

        if self._player_hp_pct(pyboy) < HP_POTION_THRESHOLD:
            self._queue_potion()
        else:
            self._queue_best_move(pyboy)

        return self._queue.pop(0) if self._queue else None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _r(self, pyboy: PyBoy, addr: int) -> int:
        return pyboy.memory[addr]

    def _r16(self, pyboy: PyBoy, addr: int) -> int:
        return (self._r(pyboy, addr) << 8) | self._r(pyboy, addr + 1)

    def _player_hp_pct(self, pyboy: PyBoy) -> float:
        hp  = self._r16(pyboy, RAM_PLAYER_HP_H)
        mhp = max(self._r16(pyboy, RAM_PLAYER_MHP_H), 1)
        return hp / mhp

    def _best_move_index(self, pyboy: PyBoy) -> int:
        enemy_types = {
            self._r(pyboy, RAM_ENEMY_TYPE1),
            self._r(pyboy, RAM_ENEMY_TYPE2),
        }
        best_idx   = 0
        best_score = -1
        for i in range(4):
            pp      = self._r(pyboy, RAM_MOVE_PP[i])
            move_id = self._r(pyboy, RAM_MOVE_IDS[i])
            if pp == 0 or move_id == 0:
                continue
            move_type = MOVE_TYPES.get(move_id, T_NORMAL)
            score = 2 if TYPE_CHART.get(move_type, set()) & enemy_types else 1
            if score > best_score:
                best_score = score
                best_idx   = i
        return best_idx

    def _queue_best_move(self, pyboy: PyBoy):
        """
        Gen 1 battle menu navigation:
          Main menu: FIGHT / ITEM / PKMN / RUN  (2×2 grid, cursor starts at FIGHT)
          Move menu: 4 moves in 2×2 grid

        Press A to select FIGHT, navigate to move slot, press A.
        """
        idx = self._best_move_index(pyboy)
        seq = ['a']                     # Open FIGHT menu
        # Move cursor within 2×2 move grid (0=top-left, 1=bottom-left, 2=top-right, 3=bottom-right)
        if idx == 1:
            seq.append('down')
        elif idx == 2:
            seq.append('right')
        elif idx == 3:
            seq += ['down', 'right']
        seq.append('a')                 # Confirm move
        self._queue = seq

    def _queue_potion(self):
        """
        Navigate Gen 1 battle menu to use a Potion:
          Main menu cursor starts at FIGHT → press DOWN to reach ITEM → A
          Item list: navigate to Potion → A → select Pokémon → A
        Simplified: assumes Potion is first item in bag.
        """
        self._queue = ['down', 'a', 'a', 'a']   # ITEM → open bag → select first item → confirm Pokémon
