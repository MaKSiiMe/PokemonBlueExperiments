"""Lecture RAM sur savestates réels : valide les adresses corrigées (audit §2.1).

Nécessite la ROM et les savestates de `states/` (sautés sinon). Les combats sont
déclenchés de façon déterministe en marchant dans l'herbe depuis un savestate.
"""

import pytest
from pyboy import PyBoy

from pokeblue.knowledge.gen1_data import MAPS, MOVE_IDS, MOVES, SPECIES
from pokeblue.state import ram_symbols as sym

FACINGS = {sym.SPRITE_FACING_DOWN, sym.SPRITE_FACING_UP, sym.SPRITE_FACING_LEFT,
           sym.SPRITE_FACING_RIGHT}
BAG_END = 0xFF  # terminateur des listes d'objets


@pytest.fixture
def pyboy(rom_path):
    pb = PyBoy(rom_path, window="null", sound=False)
    pb.set_emulation_speed(0)
    yield pb
    pb.stop()


def _load(pb, path):
    with open(path, "rb") as f:
        pb.load_state(f)
    pb.tick(2, render=False)


def _u16(pb, addr):
    return (pb.memory[addr] << 8) | pb.memory[addr + 1]


def _walk_until_battle(pb, max_steps=400):
    """Alterne haut/bas dans l'herbe jusqu'à une rencontre, puis passe l'intro du combat
    jusqu'à l'envoi du Pokémon du joueur (wBattleMon chargé)."""
    for step in range(max_steps):
        btn = ("up", "down")[(step // 2) % 2]
        pb.button_press(btn)
        pb.tick(16, render=False)
        pb.button_release(btn)
        pb.tick(8, render=False)
        if pb.memory[sym.W_IS_IN_BATTLE]:
            break
    else:
        pytest.fail("aucune rencontre déclenchée")
    # « Un X sauvage apparaît ! » attend A ; le Pokémon du joueur est envoyé ensuite.
    for _ in range(20):
        pb.tick(120, render=False)
        if pb.memory[sym.W_BATTLE_MON_SPECIES]:
            pb.tick(240, render=False)  # menu de combat affiché
            return
        pb.button("a")
    pytest.fail("le Pokémon du joueur n'a pas été envoyé")


@pytest.mark.parametrize("state", [
    "00_pallet_town.state", "21_viridian_gym_front.state", "47_pewter_gym.state",
])
def test_overworld_reads_are_consistent(pyboy, state_path, state):
    _load(pyboy, state_path(state))
    m = pyboy.memory
    assert m[sym.W_IS_IN_BATTLE] == 0
    assert m[sym.W_CUR_MAP] in MAPS
    assert m[sym.W_SPRITE_PLAYER_STATE_DATA1_FACING_DIRECTION] in FACINGS
    assert 1 <= m[sym.W_PARTY_COUNT] <= sym.PARTY_LENGTH
    assert m[sym.W_PARTY_MON1_SPECIES] in SPECIES
    assert 1 <= m[sym.W_PARTY_MON1_LEVEL] <= 100
    assert 0 < _u16(pyboy, sym.W_PARTY_MON1_HP) <= _u16(pyboy, sym.W_PARTY_MON1_MAX_HP)


def test_bag_count_matches_terminated_list(pyboy, state_path):
    _load(pyboy, state_path("30_viridian_forest_grass2.state"))
    m = pyboy.memory
    count = m[sym.W_NUM_BAG_ITEMS]
    assert 0 < count <= sym.BAG_ITEM_CAPACITY
    assert m[sym.W_BAG_ITEMS + 2 * count] == BAG_END
    assert all(m[sym.W_BAG_ITEMS + 2 * i] != BAG_END for i in range(count))


def test_event_flags_beyond_first_32_bytes(pyboy, state_path):
    """Après la Forêt de Jade, des drapeaux sont actifs au-delà des 32 octets lus avant."""
    _load(pyboy, state_path("32_viridian_forest_battle3.state"))
    flags = [pyboy.memory[sym.W_EVENT_FLAGS + i] for i in range(sym.EVENT_FLAGS_SIZE)]
    assert any(flags[32:])


def test_wild_battle_structs(pyboy, state_path):
    _load(pyboy, state_path("27_viridian_forest_grass1.state"))
    _walk_until_battle(pyboy)
    m = pyboy.memory
    assert m[sym.W_IS_IN_BATTLE] == sym.WILD_BATTLE

    enemy = m[sym.W_ENEMY_MON_SPECIES]
    assert enemy in SPECIES
    # Les types lus en RAM sont ceux de l'espèce dans les données pokered.
    assert (m[sym.W_ENEMY_MON_TYPE1], m[sym.W_ENEMY_MON_TYPE2]) == SPECIES[enemy].types
    assert 1 <= m[sym.W_ENEMY_MON_LEVEL] <= 100
    assert 0 < _u16(pyboy, sym.W_ENEMY_MON_HP) <= _u16(pyboy, sym.W_ENEMY_MON_MAX_HP)

    # Le Pokémon actif est le n°1 de l'équipe en début de combat.
    assert m[sym.W_BATTLE_MON_SPECIES] == m[sym.W_PARTY_MON1_SPECIES]
    moves = [m[sym.W_BATTLE_MON_MOVES + i] for i in range(sym.NUM_MOVES)]
    assert moves == [m[sym.W_PARTY_MON1_MOVES + i] for i in range(sym.NUM_MOVES)]
    for move, i in zip(moves, range(sym.NUM_MOVES), strict=True):
        pp = m[sym.W_BATTLE_MON_PP + i] & sym.PP_MASK
        if move:
            assert move in MOVES and 0 < pp <= MOVES[move].pp + MOVES[move].pp // 5 * 3
        else:
            assert pp == 0


def test_battle_agent_prefers_super_effective_move(pyboy, state_path):
    """Salamèche (Griffe, Rugissement, Flammèche) contre Chrysacier (Insecte) : Flammèche."""
    from src.agent.battle_agent import BattleAgent

    _load(pyboy, state_path("27_viridian_forest_grass1.state"))
    _walk_until_battle(pyboy)
    moves = [pyboy.memory[sym.W_BATTLE_MON_MOVES + i] for i in range(sym.NUM_MOVES)]
    assert MOVE_IDS["EMBER"] in moves

    agent = BattleAgent()
    assert agent._best_move_index(pyboy) == moves.index(MOVE_IDS["EMBER"])
