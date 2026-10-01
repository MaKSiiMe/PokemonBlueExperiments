"""Cohérence de `ram_symbols` (généré) : la disposition mémoire doit correspondre
aux structures déclarées dans pokered. Aucune adresse n'est écrite ici en dur :
on vérifie des relations entre symboles et constantes."""

from pokeblue.state import ram_symbols as sym


def _next_symbol_after(addr: int) -> int:
    return min(a for a in sym.SYMBOLS.values() if a > addr)


def test_symbols_table_matches_constants():
    assert sym.SYMBOLS["wEventFlags"] == sym.W_EVENT_FLAGS
    assert sym.SYMBOLS["wSpritePlayerStateData1FacingDirection"] == (
        sym.W_SPRITE_PLAYER_STATE_DATA1_FACING_DIRECTION)
    assert len(sym.SYMBOLS) > 2000


def test_event_flags_cover_all_2560_flags():
    # Audit §2.1 : l'ancien code ne lisait que 32 des 320 octets.
    assert sym.NUM_EVENTS == 2560
    assert sym.EVENT_FLAGS_SIZE == 320
    assert _next_symbol_after(sym.W_EVENT_FLAGS) - sym.W_EVENT_FLAGS == sym.EVENT_FLAGS_SIZE


def test_party_struct_layout():
    party = [getattr(sym, f"W_PARTY_MON{i}") for i in range(1, sym.PARTY_LENGTH + 1)]
    assert [b - a for a, b in zip(party, party[1:], strict=False)] == [sym.PARTYMON_STRUCT_LENGTH] * 5
    base = sym.W_PARTY_MON1
    assert sym.W_PARTY_MON1_SPECIES - base == sym.MON_SPECIES
    assert sym.W_PARTY_MON1_HP - base == sym.MON_HP
    assert sym.W_PARTY_MON1_STATUS - base == sym.MON_STATUS
    assert sym.W_PARTY_MON1_MOVES - base == sym.MON_MOVES
    assert sym.W_PARTY_MON1_PP - base == sym.MON_PP
    assert sym.W_PARTY_MON1_LEVEL - base == sym.MON_LEVEL
    assert sym.W_PARTY_MON1_MAX_HP - base == sym.MON_MAXHP
    assert sym.W_PARTY_MON6_LEVEL - sym.W_PARTY_MON6 == sym.MON_LEVEL


def test_enemy_battle_struct_is_contiguous():
    # Audit §2.1 : espèce, PV, types et niveau ennemis étaient lus à de mauvais offsets.
    assert sym.W_ENEMY_MON == sym.W_ENEMY_MON_SPECIES
    assert sym.W_ENEMY_MON_HP == sym.W_ENEMY_MON_SPECIES + 1
    assert sym.W_ENEMY_MON_TYPE2 == sym.W_ENEMY_MON_TYPE1 + 1
    assert sym.W_ENEMY_MON_MOVES - sym.W_ENEMY_MON_SPECIES == sym.W_BATTLE_MON_MOVES - sym.W_BATTLE_MON
    assert sym.W_ENEMY_MON_LEVEL - sym.W_ENEMY_MON == sym.W_BATTLE_MON_LEVEL - sym.W_BATTLE_MON
    # Les 4 moves du Pokémon actif précèdent directement ses DVs : l'ancien
    # RAM_ENEMY_TYPE1/2 lisait le 4ᵉ move et les DVs du Pokémon *joueur*.
    assert sym.W_BATTLE_MON_DVS == sym.W_BATTLE_MON_MOVES + sym.NUM_MOVES


def test_bag_and_flag_arrays():
    assert sym.W_BAG_ITEMS == sym.W_NUM_BAG_ITEMS + 1
    assert sym.W_POKEDEX_OWNED_END - sym.W_POKEDEX_OWNED == sym.POKEDEX_FLAGS_SIZE
    assert sym.W_POKEDEX_SEEN_END - sym.W_POKEDEX_SEEN == sym.POKEDEX_FLAGS_SIZE
    assert sym.BADGE_FLAGS_SIZE == 1


def test_pp_masks_partition_the_byte():
    assert sym.PP_MASK | sym.PP_UP_MASK == 0xFF
    assert sym.PP_MASK & sym.PP_UP_MASK == 0
