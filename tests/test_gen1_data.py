"""Données Gen 1 générées depuis pokered : cas connus et bizarreries de la Gen 1."""

import pytest

from pokeblue.knowledge.gen1_data import (
    DEX_TO_SPECIES,
    MAP_IDS,
    MOVE_IDS,
    MOVES,
    SPECIES,
    SPECIES_IDS,
    TYPE_IDS,
    is_damaging,
    is_special_type,
    move_priority,
    type_multiplier,
)

T = TYPE_IDS


def _types(species: str) -> tuple[int, int]:
    return SPECIES[SPECIES_IDS[species]].types


@pytest.mark.parametrize(("atk", "defender", "expected"), [
    ("GHOST", ("PSYCHIC_TYPE",), 0.0),     # bug Gen 1 : Spectre sans effet sur Psy
    ("BUG", ("POISON",), 2.0),             # Gen 1 uniquement
    ("POISON", ("BUG",), 2.0),             # Gen 1 uniquement
    ("BUG", ("PSYCHIC_TYPE",), 2.0),
    ("ICE", ("FIRE",), 1.0),               # Feu ne résiste à Glace qu'à partir de la Gen 2
    ("FIRE", ("ICE",), 2.0),
    ("PSYCHIC_TYPE", ("PSYCHIC_TYPE",), 0.5),
    ("NORMAL", ("GHOST",), 0.0),
    ("GROUND", ("FLYING",), 0.0),
    ("DRAGON", ("DRAGON",), 2.0),
])
def test_gen1_type_chart(atk, defender, expected):
    assert type_multiplier(T[atk], [T[d] for d in defender]) == expected


def test_mono_type_stored_twice_is_not_squared():
    # En RAM, Salamèche est Feu/Feu : Eau doit rester ×2, pas ×4.
    assert _types("CHARMANDER") == (T["FIRE"], T["FIRE"])
    assert type_multiplier(T["WATER"], _types("CHARMANDER")) == 2.0


def test_dual_types_multiply():
    assert type_multiplier(T["WATER"], _types("GEODUDE")) == 4.0      # Roche/Sol
    assert type_multiplier(T["ELECTRIC"], _types("GYARADOS")) == 4.0  # Eau/Vol
    assert type_multiplier(T["GROUND"], _types("CHARIZARD")) == 0.0   # Feu/Vol
    assert type_multiplier(T["GHOST"], _types("GENGAR")) == 2.0       # Spectre/Poison


@pytest.mark.parametrize("move", ["BITE", "GUST", "KARATE_CHOP"])
def test_moves_that_were_normal_in_gen1(move):
    assert MOVES[MOVE_IDS[move]].type == T["NORMAL"]


def test_move_table_values():
    tackle = MOVES[MOVE_IDS["TACKLE"]]
    assert (tackle.power, tackle.accuracy, tackle.pp) == (35, 95, 35)
    assert len(MOVES) == 165
    assert is_damaging(MOVE_IDS["SONICBOOM"])            # dégâts fixes, puissance 1
    assert not is_damaging(MOVE_IDS["SWORDS_DANCE"])


def test_only_quick_attack_and_counter_have_priority():
    assert move_priority(MOVE_IDS["QUICK_ATTACK"]) == 1
    assert move_priority(MOVE_IDS["COUNTER"]) == -1
    for move in ("METRONOME", "BIDE", "RAGE", "TACKLE"):
        assert move_priority(MOVE_IDS[move]) == 0


def test_species_internal_ids_map_one_to_one_to_dex():
    assert len(SPECIES) == 151
    assert sorted(s.dex for s in SPECIES.values()) == list(range(1, 152))
    # Les IDs internes ne suivent pas le Pokédex : Rhinoféros est le n°1 interne.
    assert SPECIES_IDS["RHYDON"] == 1
    assert SPECIES[DEX_TO_SPECIES[1]].name == "BULBASAUR"


def test_species_use_gen1_types_and_stats():
    assert _types("CLEFAIRY") == (T["NORMAL"], T["NORMAL"])         # Fée n'existe pas
    assert _types("MAGNEMITE") == (T["ELECTRIC"], T["ELECTRIC"])    # Acier n'existe pas
    mewtwo = SPECIES[SPECIES_IDS["MEWTWO"]]
    assert mewtwo.special == 154                                     # stat Spécial unique


def test_physical_special_split_by_type():
    assert is_special_type(T["FIRE"])
    assert is_special_type(T["PSYCHIC_TYPE"])
    assert not is_special_type(T["NORMAL"])
    assert not is_special_type(T["GHOST"])


def test_map_ids():
    assert MAP_IDS["PALLET_TOWN"] == 0
    assert MAP_IDS["MT_MOON_1F"] != MAP_IDS["VERMILION_POKECENTER"]
    assert {"INDIGO_PLATEAU", "LORELEIS_ROOM", "CHAMPIONS_ROOM", "HALL_OF_FAME"} <= set(MAP_IDS)
