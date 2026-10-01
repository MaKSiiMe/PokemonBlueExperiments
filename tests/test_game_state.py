"""Décodage de GameState sur une mémoire synthétique (sans ROM) et StateDiff."""

import dataclasses

import pytest

from pokeblue.knowledge.gen1_data import (
    EVENT_IDS,
    ITEM_IDS,
    MAP_IDS,
    MOVE_IDS,
    SPECIES_IDS,
    TYPE_IDS,
)
from pokeblue.state import ram_symbols as sym
from pokeblue.state.game_state import (
    GameState,
    StateDiff,
    decode_bcd,
    decode_flags,
    set_bits,
)
from pokeblue.state.memory import MemorySnapshot


def _u16(values: dict, addr: int, value: int) -> None:
    values[addr], values[addr + 1] = value >> 8, value & 0xFF


def _party_mon(values: dict, index: int, species: str, level: int, hp: int, max_hp: int,
               moves: tuple[str, ...], pp: tuple[int, ...]) -> None:
    base = sym.W_PARTY_MON1 + index * sym.PARTYMON_STRUCT_LENGTH
    values[base + sym.MON_SPECIES] = SPECIES_IDS[species]
    values[base + sym.MON_LEVEL] = level
    _u16(values, base + sym.MON_HP, hp)
    _u16(values, base + sym.MON_MAXHP, max_hp)
    for i, move in enumerate(moves):
        values[base + sym.MON_MOVES + i] = MOVE_IDS[move]
        values[base + sym.MON_PP + i] = pp[i]
    for i, byte in enumerate((0x00, 0x05, 0x39)):           # 1337 points d'expérience
        values[base + sym.MON_EXP + i] = byte


def _memory(overrides: dict | None = None) -> MemorySnapshot:
    values: dict = {
        sym.W_CUR_MAP: MAP_IDS["PEWTER_CITY"],
        sym.W_X_COORD: 25,
        sym.W_Y_COORD: 26,
        sym.W_SPRITE_PLAYER_STATE_DATA1_FACING_DIRECTION: sym.SPRITE_FACING_UP,
        sym.W_PARTY_COUNT: 2,
        sym.W_NUM_BAG_ITEMS: 2,
        sym.W_BAG_ITEMS: ITEM_IDS["POTION"], sym.W_BAG_ITEMS + 1: 2,
        sym.W_BAG_ITEMS + 2: ITEM_IDS["POKE_BALL"], sym.W_BAG_ITEMS + 3: 5,
        sym.W_BAG_ITEMS + 4: 0xFF,
        sym.W_PLAYER_MONEY: 0x00, sym.W_PLAYER_MONEY + 1: 0x33, sym.W_PLAYER_MONEY + 2: 0x95,
        sym.W_OBTAINED_BADGES: 1 << sym.BIT_BOULDERBADGE,
        sym.R_LCDC: 1 << sym.B_LCDC_ENABLE,
        sym.R_BGP: 0xE4,
    }
    beat_brock = EVENT_IDS["EVENT_BEAT_BROCK"]
    values[sym.W_EVENT_FLAGS + beat_brock // 8] = 1 << (beat_brock % 8)
    values[sym.W_POKEDEX_OWNED] = 0b1000          # n° 4 (Salamèche)
    values[sym.W_POKEDEX_SEEN] = 0b1001           # n° 1 et 4
    _party_mon(values, 0, "CHARMANDER", 12, 35, 35, ("SCRATCH", "GROWL", "EMBER"), (30, 40, 25))
    # 2e Pokémon : PP du 1er move avec 2 PP Plus (bits de poids fort)
    _party_mon(values, 1, "PIDGEY", 5, 0, 19, ("GUST",), (0b1000_0000 | 33,))
    values.update(overrides or {})
    return MemorySnapshot.blank(values)


def test_decode_bcd():
    assert decode_bcd(bytes([0x00, 0x33, 0x95])) == 3395
    assert decode_bcd(bytes([0x99, 0x99, 0x99])) == 999_999
    with pytest.raises(ValueError):
        decode_bcd(bytes([0x0A]))


def test_flag_arrays_are_little_endian_bitsets():
    # Drapeau 9 = bit 1 de l'octet 1 ; drapeau 0 = bit 0 de l'octet 0.
    assert decode_flags(bytes([0b0000_0001, 0b0000_0010])) == (1 << 0) | (1 << 9)
    assert set_bits((1 << 0) | (1 << 9) | (1 << 2559)) == (0, 9, 2559)
    assert set_bits(0) == ()


def test_position_money_badges_and_flags():
    state = GameState.from_memory(_memory())
    assert (state.map_id, state.x, state.y) == (MAP_IDS["PEWTER_CITY"], 25, 26)
    assert state.facing == sym.SPRITE_FACING_UP
    assert state.money == 3395
    assert state.has_badge(sym.BIT_BOULDERBADGE) and state.n_badges == 1
    assert state.flag(EVENT_IDS["EVENT_BEAT_BROCK"]) and state.n_flags == 1
    assert state.owns(4) and not state.owns(1) and state.has_seen(1)


def test_party_struct_decoding():
    state = GameState.from_memory(_memory())
    charmander, pidgey = state.party
    assert charmander.species == SPECIES_IDS["CHARMANDER"]
    assert (charmander.level, charmander.hp, charmander.max_hp) == (12, 35, 35)
    assert charmander.moves == (MOVE_IDS["SCRATCH"], MOVE_IDS["GROWL"], MOVE_IDS["EMBER"], 0)
    assert charmander.pp == (30, 40, 25, 0)
    assert charmander.exp == 1337
    assert pidgey.pp[0] == 33          # PP Plus masqués
    assert pidgey.fainted and not state.all_fainted


def test_bag_is_limited_to_its_count():
    state = GameState.from_memory(_memory())
    assert state.bag == ((ITEM_IDS["POTION"], 2), (ITEM_IDS["POKE_BALL"], 5))
    assert state.item_count(ITEM_IDS["POKE_BALL"]) == 5
    assert state.item_count(ITEM_IDS["ANTIDOTE"]) == 0


def test_battle_structs():
    overrides = {
        sym.W_IS_IN_BATTLE: sym.WILD_BATTLE,
        sym.W_ENEMY_MON_SPECIES: SPECIES_IDS["METAPOD"],
        sym.W_ENEMY_MON_LEVEL: 6,
        sym.W_ENEMY_MON_TYPE1: TYPE_IDS["BUG"],
        sym.W_ENEMY_MON_TYPE2: TYPE_IDS["BUG"],
        sym.W_ENEMY_MON_HP: 0, sym.W_ENEMY_MON_HP + 1: 21,
        sym.W_ENEMY_MON_MAX_HP: 0, sym.W_ENEMY_MON_MAX_HP + 1: 21,
        **{sym.W_ENEMY_MON_STAT_MODS + i: sym.BASE_STAT_LEVEL for i in range(6)},
        sym.W_ENEMY_MON_DEFENSE_MOD: sym.BASE_STAT_LEVEL + 1,   # Armure utilisée une fois
    }
    battle = GameState.from_memory(_memory(overrides)).battle
    assert battle is not None and battle.is_wild
    assert battle.enemy.species == SPECIES_IDS["METAPOD"]
    assert battle.enemy.types == (TYPE_IDS["BUG"], TYPE_IDS["BUG"])
    assert (battle.enemy.hp, battle.enemy.max_hp, battle.enemy.level) == (21, 21, 6)
    assert battle.enemy.stat_mods == (7, 8, 7, 7, 7, 7)
    assert GameState.from_memory(_memory()).battle is None


def _with(state: GameState, **changes) -> GameState:
    return dataclasses.replace(state, **changes)


def test_diff_reports_progress_events():
    prev = GameState.from_memory(_memory())
    gym = EVENT_IDS["EVENT_GOT_TM34"]
    cur = _with(
        prev,
        map_id=MAP_IDS["PEWTER_GYM"],
        event_flags=prev.event_flags | (1 << gym),
        badges=prev.badges | (1 << sym.BIT_CASCADEBADGE),
        money=prev.money + 450,
        bag=((ITEM_IDS["POTION"], 1), (ITEM_IDS["POKE_BALL"], 5), (ITEM_IDS["TM_BIDE"], 1)),
        party=(dataclasses.replace(prev.party[0], level=13), prev.party[1]),
        pokedex_seen=prev.pokedex_seen | (1 << 94),     # Onix (n° 95)
    )
    diff = cur.diff(prev)
    assert diff.new_flags == (gym,) and not diff.cleared_flags
    assert diff.map_change == (MAP_IDS["PEWTER_CITY"], MAP_IDS["PEWTER_GYM"]) and diff.moved
    assert diff.new_badges == (sym.BIT_CASCADEBADGE,)
    assert diff.money_delta == 450
    assert diff.item_deltas == ((ITEM_IDS["POTION"], -1), (ITEM_IDS["TM_BIDE"], 1))
    assert diff.level_ups == ((0, 12, 13),)
    assert diff.new_seen == (95,) and not diff.new_owned
    lines = diff.describe()
    assert "+ EVENT_GOT_TM34" in lines and "carte PEWTER_CITY → PEWTER_GYM" in lines
    assert "vu ONIX" in lines


def test_diff_faint_and_battle_events():
    calm = GameState.from_memory(_memory())
    fighting = GameState.from_memory(_memory({
        sym.W_IS_IN_BATTLE: sym.TRAINER_BATTLE,
        sym.W_ENEMY_MON_SPECIES: SPECIES_IDS["ONIX"], sym.W_ENEMY_MON_LEVEL: 14,
        sym.W_ENEMY_MON_HP + 1: 10,
    }))
    enemy_down = GameState.from_memory(_memory({
        sym.W_IS_IN_BATTLE: sym.TRAINER_BATTLE,
        sym.W_ENEMY_MON_SPECIES: SPECIES_IDS["ONIX"], sym.W_ENEMY_MON_LEVEL: 14,
    }))
    start, ko = fighting.diff(calm), enemy_down.diff(fighting)
    assert start.battle_started and not start.battle_ended
    assert ko.enemy_fainted and not ko.battle_started
    assert calm.diff(enemy_down).battle_ended

    lead_down = _with(calm, party=(dataclasses.replace(calm.party[0], hp=0), calm.party[1]))
    assert lead_down.diff(calm).fainted == (0,)
    assert lead_down.all_fainted


def test_empty_diff_is_falsy():
    state = GameState.from_memory(_memory())
    assert not state.diff(state)
    assert isinstance(state.diff(state), StateDiff)
