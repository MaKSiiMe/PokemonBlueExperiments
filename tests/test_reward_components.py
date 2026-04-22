"""
test_reward_components.py — Vérifie chaque composante de reward isolément.
"""

import pytest


def _call_reward(env):
    """Appelle _reward() avec la position courante et retourne (reward, components)."""
    from src.emulator.ram_map import RAM_PLAYER_X, RAM_PLAYER_Y, RAM_MAP_ID
    x   = env._r(RAM_PLAYER_X)
    y   = env._r(RAM_PLAYER_Y)
    mid = env._r(RAM_MAP_ID)
    reward = env._reward(x, y, mid)
    return reward, dict(env._r_components)


def test_r_tile_new_tile(env):
    """Première visite d'une tile jamais vue doit donner r_tile = +1.0."""
    env._seen_tiles.clear()
    env._seen_arrays.clear()
    _, comps = _call_reward(env)
    assert comps['r_tile'] == pytest.approx(1.0), (
        f"r_tile attendu 1.0, obtenu {comps['r_tile']}"
    )


def test_r_tile_revisit_no_bonus(env):
    """Revisiter une tile connue ne donne pas de bonus tile."""
    _call_reward(env)
    _, comps = _call_reward(env)
    assert comps['r_tile'] == pytest.approx(0.0), (
        f"r_tile attendu 0.0 sur revisit, obtenu {comps['r_tile']}"
    )


def test_r_map_new_map(env):
    """Entrer dans une nouvelle map doit déclencher le bonus r_map."""
    from src.emulator.ram_map import RAM_MAP_ID
    current_map = env._r(RAM_MAP_ID)
    fake_prev   = (current_map + 1) % 255
    env._prev_map_id = fake_prev
    env._visited_maps.discard(current_map)

    _, comps = _call_reward(env)
    assert comps['r_map'] > 0.0, f"r_map attendu > 0, obtenu {comps['r_map']}"


def test_reward_is_sum_of_components(env):
    """Le reward total doit être la somme exacte des composantes."""
    reward, comps = _call_reward(env)
    expected = sum(comps.values())
    assert reward == pytest.approx(expected, abs=1e-6), (
        f"reward={reward} != sum(components)={expected}"
    )


def test_r_components_keys(env):
    """_r_components doit contenir exactement les 8 clés attendues."""
    _call_reward(env)
    assert set(env._r_components.keys()) == {
        'r_map', 'r_tile', 'r_event', 'r_type', 'r_victory',
        'r_level', 'r_stuck', 'r_progress',
    }
