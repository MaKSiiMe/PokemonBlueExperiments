"""Structures des tables Gen 1 générées (`tables.py`)."""

from __future__ import annotations

from typing import NamedTuple


class Move(NamedTuple):
    """Une attaque (data/moves/moves.asm)."""

    name: str      # constante pokered, ex. "KARATE_CHOP"
    effect: str    # constante d'effet (constants/move_effect_constants.asm)
    power: int     # 0 = attaque sans dégâts directs ; 1 = dégâts fixes/spéciaux (Frappe Atlas…)
    type: int      # octet de type (voir TYPE_NAMES)
    accuracy: int  # pourcentage écrit dans moves.asm ; le jeu stocke accuracy * 255 // 100
    pp: int


class Species(NamedTuple):
    """Une espèce (data/pokemon/base_stats/*.asm), indexée par son ID interne."""

    name: str                     # constante pokered, ex. "NIDORAN_M"
    dex: int                      # numéro du Pokédex national
    hp: int
    attack: int
    defense: int
    speed: int
    special: int                  # stat Spécial unique de la Gen 1
    types: tuple[int, int]        # un mono-type répète son type (comme en RAM)
    catch_rate: int
    base_exp: int
    start_moves: tuple[int, ...]  # attaques connues au niveau 1 (sans NO_MOVE)
    growth_rate: str              # constante pokered, ex. "GROWTH_MEDIUM_SLOW"


class MapInfo(NamedTuple):
    """Une carte (constants/map_constants.asm). Dimensions en blocs de 2×2 cases."""

    name: str
    width: int
    height: int
