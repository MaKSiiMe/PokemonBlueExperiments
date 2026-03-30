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
    RAM_ENEMY_HP_H, RAM_ENEMY_HP_L, RAM_ENEMY_MHP_H, RAM_ENEMY_MHP_L,
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

# Tous les moves Gen 1 — IDs depuis pret/pokered (1-indexés, hex).
# Note Gen 1 : Karate Chop, Gust, Bite sont Normal (pas Fighting/Flying/Dark).
MOVE_TYPES: dict[int, int] = {
    # Normal
    0x01: T_NORMAL,  # Pound
    0x02: T_NORMAL,  # Karate Chop  (Normal en Gen 1 !)
    0x03: T_NORMAL,  # Double Slap
    0x04: T_NORMAL,  # Comet Punch
    0x05: T_NORMAL,  # Mega Punch
    0x06: T_NORMAL,  # Pay Day
    0x09: T_ELECTRIC,# ThunderPunch  — électrique pas normal
    0x07: T_FIRE,    # Fire Punch
    0x08: T_ICE,     # Ice Punch
    0x0A: T_NORMAL,  # Scratch
    0x0B: T_NORMAL,  # ViceGrip
    0x0C: T_NORMAL,  # Guillotine (OHKO)
    0x0D: T_NORMAL,  # Razor Wind
    0x0F: T_NORMAL,  # Cut
    0x10: T_NORMAL,  # Gust         (Normal en Gen 1 !)
    0x11: T_FLYING,  # Wing Attack
    0x13: T_FLYING,  # Fly
    0x14: T_NORMAL,  # Bind
    0x15: T_NORMAL,  # Slam
    0x16: T_GRASS,   # Vine Whip
    0x17: T_NORMAL,  # Stomp
    0x18: T_FIGHT,   # Double Kick
    0x19: T_NORMAL,  # Mega Kick
    0x1A: T_FIGHT,   # Jump Kick
    0x1B: T_FIGHT,   # Rolling Kick
    0x1D: T_NORMAL,  # Headbutt
    0x1E: T_NORMAL,  # Horn Attack
    0x1F: T_NORMAL,  # Fury Attack
    0x20: T_NORMAL,  # Horn Drill (OHKO)
    0x21: T_NORMAL,  # Tackle
    0x22: T_NORMAL,  # Body Slam
    0x23: T_NORMAL,  # Wrap
    0x24: T_NORMAL,  # Take Down
    0x25: T_NORMAL,  # Thrash
    0x26: T_NORMAL,  # Double-Edge
    0x28: T_POISON,  # Poison Sting
    0x29: T_BUG,     # Twineedle
    0x2A: T_BUG,     # Pin Missile
    0x2C: T_NORMAL,  # Bite         (Normal en Gen 1 !)
    0x31: T_NORMAL,  # SonicBoom
    0x33: T_POISON,  # Acid
    0x34: T_FIRE,    # Ember
    0x35: T_FIRE,    # Flamethrower
    0x37: T_WATER,   # Water Gun
    0x38: T_WATER,   # Hydro Pump
    0x39: T_WATER,   # Surf
    0x3A: T_ICE,     # Ice Beam
    0x3B: T_ICE,     # Blizzard
    0x3C: T_PSYCHIC, # Psybeam
    0x3D: T_WATER,   # BubbleBeam
    0x3E: T_ICE,     # Aurora Beam
    0x3F: T_NORMAL,  # Hyper Beam
    0x40: T_FLYING,  # Peck
    0x41: T_FLYING,  # Drill Peck
    0x42: T_FIGHT,   # Submission
    0x43: T_FIGHT,   # Low Kick
    0x44: T_FIGHT,   # Counter
    0x45: T_FIGHT,   # Seismic Toss
    0x46: T_NORMAL,  # Strength
    0x47: T_GRASS,   # Absorb
    0x48: T_GRASS,   # Mega Drain
    0x4B: T_GRASS,   # Razor Leaf
    0x4C: T_GRASS,   # SolarBeam
    0x50: T_GRASS,   # Petal Dance
    0x51: T_BUG,     # String Shot
    0x52: T_DRAGON,  # Dragon Rage
    0x53: T_FIRE,    # Fire Spin
    0x54: T_ELECTRIC,# ThunderShock
    0x55: T_ELECTRIC,# Thunderbolt
    0x57: T_ELECTRIC,# Thunder
    0x58: T_ROCK,    # Rock Throw
    0x59: T_GROUND,  # Earthquake
    0x5A: T_GROUND,  # Fissure (OHKO)
    0x5B: T_GROUND,  # Dig
    0x5D: T_PSYCHIC, # Confusion
    0x5E: T_PSYCHIC, # Psychic
    0x62: T_NORMAL,  # Quick Attack
    0x63: T_NORMAL,  # Rage
    0x65: T_GHOST,   # Night Shade
    0x6A: T_GHOST,   # Confuse Ray
    0x75: T_NORMAL,  # Bide
    0x76: T_NORMAL,  # Metronome
    0x77: T_FLYING,  # Mirror Move
    0x78: T_NORMAL,  # Self-Destruct
    0x79: T_NORMAL,  # Egg Bomb
    0x7A: T_GHOST,   # Lick
    0x7B: T_POISON,  # Smog
    0x7C: T_POISON,  # Sludge
    0x7D: T_GROUND,  # Bone Club
    0x7E: T_FIRE,    # Fire Blast
    0x7F: T_WATER,   # Waterfall
    0x80: T_WATER,   # Clamp
    0x81: T_NORMAL,  # Swift
    0x82: T_NORMAL,  # Skull Bash
    0x83: T_NORMAL,  # Spike Cannon
    0x84: T_NORMAL,  # Constrict
    0x88: T_FIGHT,   # Hi Jump Kick
    0x8A: T_PSYCHIC, # Dream Eater
    0x8D: T_BUG,     # Leech Life
    0x8F: T_FLYING,  # Sky Attack
    0x91: T_WATER,   # Bubble
    0x95: T_PSYCHIC, # Psywave
    0x98: T_WATER,   # Crabhammer
    0x99: T_NORMAL,  # Explosion
    0x9A: T_NORMAL,  # Fury Swipes
    0x9B: T_GROUND,  # Bonemerang
    0x9D: T_ROCK,    # Rock Slide
    0x9E: T_NORMAL,  # Hyper Fang
    0xA1: T_NORMAL,  # Tri Attack
    0xA2: T_NORMAL,  # Super Fang
    0xA3: T_NORMAL,  # Slash
    0xA5: T_NORMAL,  # Struggle
}

# Moves qui ne font pas de dégâts directs — à éviter si un move offensif existe.
STATUS_MOVES: set[int] = {
    0x0E,  # Swords Dance
    0x12,  # Whirlwind
    0x1C,  # Sand-Attack
    0x27,  # Tail Whip
    0x2B,  # Leer
    0x2D,  # Growl
    0x2E,  # Roar
    0x2F,  # Sing
    0x30,  # Supersonic
    0x32,  # Disable
    0x36,  # Mist
    0x49,  # Leech Seed
    0x4A,  # Growth
    0x4D,  # PoisonPowder
    0x4E,  # Stun Spore
    0x4F,  # Sleep Powder
    0x56,  # Thunder Wave
    0x5C,  # Toxic
    0x5F,  # Hypnosis
    0x60,  # Meditate
    0x61,  # Agility
    0x64,  # Teleport
    0x66,  # Mimic
    0x67,  # Screech
    0x68,  # Double Team
    0x69,  # Recover
    0x6B,  # Minimize
    0x6C,  # Smokescreen
    0x6D,  # Confuse Ray
    0x6E,  # Withdraw
    0x6F,  # Defense Curl
    0x70,  # Barrier
    0x71,  # Light Screen
    0x72,  # Haze
    0x73,  # Reflect
    0x74,  # Focus Energy
    0x85,  # Amnesia
    0x86,  # Kinesis
    0x87,  # Soft-Boiled
    0x89,  # Glare
    0x8B,  # Poison Gas
    0x8E,  # Lovely Kiss
    0x90,  # Transform
    0x93,  # Spore
    0x94,  # Flash
    0x96,  # Splash
    0x97,  # Acid Armor
    0x9C,  # Rest
    0x9F,  # Sharpen
    0xA0,  # Conversion
    0xA4,  # Substitute
}


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

    def _enemy_hp_pct(self, pyboy: PyBoy) -> float:
        hp  = (pyboy.memory[RAM_ENEMY_HP_H]  << 8) | pyboy.memory[RAM_ENEMY_HP_L]
        mhp = (pyboy.memory[RAM_ENEMY_MHP_H] << 8) | pyboy.memory[RAM_ENEMY_MHP_L]
        return hp / mhp if mhp > 0 else 1.0

    def _best_move_index(self, pyboy: PyBoy) -> int:
        enemy_types  = {pyboy.memory[RAM_ENEMY_TYPE1], pyboy.memory[RAM_ENEMY_TYPE2]}
        enemy_low_hp = self._enemy_hp_pct(pyboy) < 0.25  # pas la peine de staller
        best_idx, best_score = 0, -1

        for i in range(4):
            pp      = pyboy.memory[RAM_MOVE_PP[i]]
            move_id = pyboy.memory[RAM_MOVE_IDS[i]]
            if pp == 0 or move_id == 0:
                continue
            # Status moves inutiles si l'ennemi est presque KO
            if move_id in STATUS_MOVES:
                score = -1 if enemy_low_hp else 0
            else:
                move_type = MOVE_TYPES.get(move_id, T_NORMAL)
                score = 2 if TYPE_CHART.get(move_type, set()) & enemy_types else 1

            if score > best_score:
                best_score, best_idx = score, i

        return best_idx
