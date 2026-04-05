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

Scoring des moves (via PokemonKnowledgeGraph) :
  - Multiplicateur de type exact Gen 1 : 0.0 / 0.5 / 1.0 / 2.0 / 4.0
    (inclut immunités et résistances, absentes de l'ancien TYPE_CHART binaire)
  - Bonus +0.5 pour Quick Attack si ennemi < 30% HP (finisher avant contre-attaque)
  - Moves de statut : score 0.0 (ou -1.0 si ennemi presque KO)
"""

from __future__ import annotations

import logging

from pyboy import PyBoy

from src.emulator.ram_map import (
    RAM_ENEMY_HP_H, RAM_ENEMY_HP_L,
    RAM_ENEMY_MHP_H, RAM_ENEMY_MHP_L,
    RAM_ENEMY_TYPE1, RAM_ENEMY_TYPE2,
    RAM_MOVE_IDS, RAM_MOVE_PP,
)
from src.knowledge import PokemonKnowledgeGraph
from src.knowledge.gen1_data import (
    MOVE_TYPES, STATUS_MOVES,
    RAM_TYPE_BYTE_TO_NAME,
)

logger = logging.getLogger(__name__)

# MOVE_TYPES et STATUS_MOVES sont maintenant dans src.knowledge.gen1_data
# (source unique de vérité, partagée avec pokemon_env.py sans circular import)

# ID du move Quick Attack en Gen 1 (priorité +1)
_QUICK_ATTACK_ID = 0x62


class BattleAgent:
    """Heuristic battle agent for Gen 1 Pokémon.

    Utilise PokemonKnowledgeGraph pour le scoring des types :
    multiplicateurs exacts Gen 1 (0.0 / 0.5 / 1.0 / 2.0 / 4.0)
    au lieu du TYPE_CHART binaire précédent.
    """

    _INTRO_PRESSES = 8    # A à spammer pour les dialogs d'intro
    _POST_PRESSES  = 12   # A après le move (animations + tour ennemi)

    def __init__(self, kg: PokemonKnowledgeGraph | None = None) -> None:
        """
        Args:
            kg: Instance de PokemonKnowledgeGraph. Si None, en crée une
                automatiquement (charge le graphe depuis le disque).
        """
        self._kg: PokemonKnowledgeGraph = kg or PokemonKnowledgeGraph()
        self._queue: list[str] = []
        self._turn: int = 0

    def reset(self) -> None:
        self._queue = ['a'] * self._INTRO_PRESSES
        self._turn  = 0

    def act(self, pyboy: PyBoy) -> str | None:
        if self._queue:
            return self._queue.pop(0)
        self._build_turn_queue(pyboy)
        return self._queue.pop(0) if self._queue else None

    def _build_turn_queue(self, pyboy: PyBoy) -> None:
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
        """Sélectionne le meilleur move via le graphe de connaissances.

        Scoring :
          • Move offensif : type_multiplier_from_ram() × le type du move
            → 0.0 immune / 0.5 résistance / 1.0 neutre / 2.0 SE / 4.0 double SE
          • Quick Attack + ennemi < 30% HP : +0.5 bonus finisher
          • Move de statut : 0.0, ou -1.0 si ennemi presque KO
        """
        enemy_type_bytes = [
            pyboy.memory[RAM_ENEMY_TYPE1],
            pyboy.memory[RAM_ENEMY_TYPE2],
        ]
        enemy_hp_pct = self._enemy_hp_pct(pyboy)
        best_idx, best_score = 0, -999.0

        for i in range(4):
            pp      = pyboy.memory[RAM_MOVE_PP[i]]
            move_id = pyboy.memory[RAM_MOVE_IDS[i]]
            if pp == 0 or move_id == 0:
                continue

            if move_id in STATUS_MOVES:
                score = -1.0 if enemy_hp_pct < 0.25 else 0.0
            else:
                move_type_byte = MOVE_TYPES.get(move_id, T_NORMAL)
                score = self._kg.type_multiplier_from_ram(move_type_byte, enemy_type_bytes)
                # Quick Attack finisher : agit avant la contre-attaque ennemie
                if move_id == _QUICK_ATTACK_ID and enemy_hp_pct < 0.3:
                    score += 0.5

            if score > best_score:
                best_score, best_idx = score, i

            logger.debug(
                "  Move slot %d : id=0x%02X  score=%.2f", i, move_id, score
            )

        return best_idx
