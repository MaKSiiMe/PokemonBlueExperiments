"""
gen1_data.py — Vues historiques (noms en minuscules façon PokéAPI) des données Gen 1.

Toutes les valeurs sont dérivées de `pokeblue.knowledge.gen1_data`, généré depuis
pret/pokered : plus aucune table n'est saisie à la main ici. Le nouveau code doit
utiliser `pokeblue.knowledge.gen1_data` directement (identifiants du jeu).

Corrections apportées par rapport à l'ancienne version manuelle :
  • Glace → Feu vaut ×1 en Gen 1 (pas ×2) ;
  • GEN1_INTERNAL_TO_DEX contenait des doublons et des IDs faux (ex. Mew, Ronflex) ;
  • seules Vive-Attaque (+1) et Riposte (−1) ont une priorité en Gen 1 ;
  • MOVE_TYPES couvre désormais les 165 attaques (plus de type par défaut).
"""

from __future__ import annotations

from pokeblue.knowledge import gen1_data as g1


def _legacy_name(type_id: int) -> str:
    """"PSYCHIC_TYPE" → "psychic" (noms PokéAPI utilisés par le graphe)."""
    return g1.TYPE_NAMES[type_id].lower().removesuffix("_type")


# ── Types ─────────────────────────────────────────────────────────────────────
# Octet RAM → nom. Inclut BIRD (0x06), type inutilisé porté seulement par MissingNo.
RAM_TYPE_BYTE_TO_NAME: dict[int, str] = {tid: _legacy_name(tid) for tid in g1.TYPE_NAMES}
RAM_TYPE_NAME_TO_BYTE: dict[str, int] = {v: k for k, v in RAM_TYPE_BYTE_TO_NAME.items()}

# Les 15 types portés par au moins une espèce (sans BIRD).
GEN1_TYPES: frozenset[str] = frozenset(
    _legacy_name(t) for species in g1.SPECIES.values() for t in species.types
)

# TYPE_CHART[attaquant][défenseur] = multiplicateur ; seules les valeurs != 1.0.
# Pour un défenseur à deux types, utiliser `type_multiplier` (pas de double comptage).
TYPE_CHART: dict[str, dict[str, float]] = {}
for _atk, _dfn, _factor in g1.TYPE_EFFECTS:
    TYPE_CHART.setdefault(_legacy_name(_atk), {})[_legacy_name(_dfn)] = _factor / 10


def type_multiplier(atk_type: str, def_types: list[str]) -> float:
    """Multiplicateur Gen 1 par noms de types ; un type répété ne compte qu'une fois."""
    mult = 1.0
    for def_type in set(def_types):
        mult *= TYPE_CHART.get(atk_type, {}).get(def_type, 1.0)
    return mult


# ── Espèces : ID interne (octet RAM) ↔ numéro de Pokédex ──────────────────────
GEN1_INTERNAL_TO_DEX: dict[int, int] = {sid: s.dex for sid, s in g1.SPECIES.items()}
GEN1_DEX_TO_INTERNAL: dict[int, int] = dict(g1.DEX_TO_SPECIES)

# ── Zones du parcours Bourg Palette → Argenta (slugs PokéAPI des rencontres) ──
_M = g1.MAP_IDS
ZONE_MAP_ID: dict[int, dict[str, str | None]] = {
    _M["PALLET_TOWN"]:     {"name": "Bourg Palette",   "pokeapi_slug": None},
    _M["VIRIDIAN_CITY"]:   {"name": "Jadielle City",   "pokeapi_slug": None},
    _M["PEWTER_CITY"]:     {"name": "Argenta City",    "pokeapi_slug": None},
    _M["ROUTE_1"]:         {"name": "Route 1",         "pokeapi_slug": "kanto-route-1-area"},
    _M["ROUTE_2"]:         {"name": "Route 2",         "pokeapi_slug": "kanto-route-2-south-towards-viridian-city"},
    # FAUX, conservé tel quel jusqu'à la refonte du graphe (Phase 2) : la carte 0x0E
    # est ROUTE_3. Les deux zones PokéAPI de la Route 2 sont sur la même carte ROUTE_2.
    _M["ROUTE_3"]:         {"name": "Route 2 Nord",    "pokeapi_slug": "kanto-route-2-north-towards-pewter-city"},
    _M["VIRIDIAN_FOREST"]: {"name": "Forêt de Jade",   "pokeapi_slug": "viridian-forest-area"},
    _M["PEWTER_GYM"]:      {"name": "Arène d'Argenta", "pokeapi_slug": None},
}

# ── Attaques ──────────────────────────────────────────────────────────────────
MOVE_TYPES: dict[int, int] = {mid: move.type for mid, move in g1.MOVES.items()}
# Attaques sans dégâts directs (puissance 0 dans moves.asm).
STATUS_MOVES: frozenset[int] = frozenset(mid for mid in g1.MOVES if not g1.is_damaging(mid))
PRIORITY_MOVES: dict[int, int] = dict(g1.MOVE_PRIORITY)
