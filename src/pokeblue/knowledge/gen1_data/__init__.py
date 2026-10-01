"""Données Gen 1 exactes (pret/pokered) et requêtes élémentaires.

Les tables viennent de `tables.py`, généré par `scripts/gen_gen1_data.py`.
Les identifiants sont ceux du jeu : un octet lu en RAM s'utilise directement
comme clé (`MOVES[move_id]`, `SPECIES[species_id]`, `TYPE_NAMES[type_id]`).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable

from pokeblue.knowledge.gen1_data.models import MapInfo, Move, Species
from pokeblue.knowledge.gen1_data.tables import (
    CHARMAP,
    EVENTS,
    EVOLUTIONS,
    FADE_PALETTES,
    ITEMS,
    LEARNSETS,
    LONE_MOVES,
    MAPS,
    MOVES,
    SOURCE_COMMIT,
    SPECIAL_TYPES_START,
    SPECIES,
    TEAM_MOVES,
    TOGGLE_NAMES,
    TRAINER_CLASSES,
    TRAINER_PARTIES,
    TYPE_EFFECTS,
    TYPE_NAMES,
)

TYPE_IDS: dict[str, int] = {name: tid for tid, name in TYPE_NAMES.items()}
MOVE_IDS: dict[str, int] = {move.name: mid for mid, move in MOVES.items()}
SPECIES_IDS: dict[str, int] = {s.name: sid for sid, s in SPECIES.items()}
MAP_IDS: dict[str, int] = {m.name: mid for mid, m in MAPS.items()}
DEX_TO_SPECIES: dict[int, int] = {s.dex: sid for sid, s in SPECIES.items()}
ITEM_IDS: dict[str, int] = {name: iid for iid, name in ITEMS.items()}
EVENT_IDS: dict[str, int] = {name: eid for eid, name in EVENTS.items()}
TOGGLE_IDS: dict[str, int] = {name: i for i, name in enumerate(TOGGLE_NAMES)}

# Tuile de police (>= 0x60) → caractère affiché par la version US. Plusieurs caractères
# partagent une tuile selon le graphisme chargé (▲ de la carte et ▶, kana japonais du
# texte non traduit) : on ignore les kana (pleine chasse) et le dernier défini l'emporte.
TILE_CHARS: dict[int, str] = {
    tile: char
    for char, tile in CHARMAP.items()
    if tile >= 0x60 and not char.startswith("<")
    and unicodedata.east_asian_width(char[0]) not in ("W", "F")
}

# Ordre d'action (engine/battle/core.asm, MainInBattleLoop) : Vive-Attaque passe avant,
# Riposte après ; sinon la Vitesse décide. Aucune autre attaque n'a de priorité en Gen 1.
MOVE_PRIORITY: dict[int, int] = {MOVE_IDS["QUICK_ATTACK"]: 1, MOVE_IDS["COUNTER"]: -1}


def type_multiplier(move_type: int, defender_types: Iterable[int]) -> float:
    """Multiplicateur d'efficacité Gen 1 d'une attaque contre un défenseur.

    Reproduit AdjustDamageForMoveType (engine/battle/core.asm) : chaque entrée de
    TypeEffects s'applique au plus une fois, si son type défenseur est l'un des types
    du défenseur. Un mono-type, stocké deux fois en RAM (ex. Feu/Feu), n'est donc
    pas compté deux fois.
    """
    defenders = set(defender_types)
    mult = 1.0
    for atk, dfn, factor in TYPE_EFFECTS:
        if atk == move_type and dfn in defenders:
            mult *= factor / 10
    return mult


def is_damaging(move_id: int) -> bool:
    """Vrai si l'attaque inflige des dégâts directs (puissance > 0 dans moves.asm)."""
    return MOVES[move_id].power > 0


def is_special_type(type_id: int) -> bool:
    """Vrai si le type utilise la stat Spécial (types >= SPECIAL dans type_constants.asm)."""
    return type_id >= SPECIAL_TYPES_START


def move_priority(move_id: int) -> int:
    return MOVE_PRIORITY.get(move_id, 0)


def decode_tiles(tiles: bytes, unknown: str = "·") -> str:
    """Texte affiché par une suite de tuiles (les tuiles de décor deviennent `unknown`)."""
    return "".join(TILE_CHARS.get(t, unknown) for t in tiles)


__all__ = [
    "CHARMAP",
    "DEX_TO_SPECIES",
    "EVENTS",
    "EVENT_IDS",
    "EVOLUTIONS",
    "FADE_PALETTES",
    "ITEMS",
    "ITEM_IDS",
    "LEARNSETS",
    "LONE_MOVES",
    "MAPS",
    "MAP_IDS",
    "MOVES",
    "MOVE_IDS",
    "MOVE_PRIORITY",
    "SOURCE_COMMIT",
    "SPECIAL_TYPES_START",
    "SPECIES",
    "SPECIES_IDS",
    "TYPE_EFFECTS",
    "TYPE_IDS",
    "TEAM_MOVES",
    "TILE_CHARS",
    "TOGGLE_IDS",
    "TOGGLE_NAMES",
    "TRAINER_CLASSES",
    "TRAINER_PARTIES",
    "TYPE_NAMES",
    "MapInfo",
    "Move",
    "Species",
    "decode_tiles",
    "is_damaging",
    "is_special_type",
    "move_priority",
    "type_multiplier",
]
