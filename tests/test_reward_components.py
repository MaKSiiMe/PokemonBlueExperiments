"""
test_reward_components.py — Vérifie chaque composante de reward isolément.

Technique : manipuler les attributs _prev_* de l'env pour forcer un delta
connu, puis appeler _reward() directement et inspecter _r_components.
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
    """Première visite d'une tile doit donner r_tile = +0.5."""
    env._seen_tiles.clear()
    env._seen_arrays.clear()
    _, comps = _call_reward(env)
    assert comps['r_tile'] == pytest.approx(0.5), (
        f"r_tile attendu 0.5, obtenu {comps['r_tile']}"
    )


def test_r_tile_revisit_no_bonus(env):
    """Revisiter une tile connue ne donne pas de bonus tile."""
    # Premier appel pour marquer la tile comme visitée
    _call_reward(env)
    # Deuxième appel : même tile, pas de bonus
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


def test_r_heal_heal_detected(env):
    """Simuler un soin : HP équipe passe de 10 à 20 en overworld."""
    from src.emulator.ram_map import RAM_BATTLE
    env.pyboy.memory[RAM_BATTLE] = 0   # overworld

    total_max = max(env._total_party_max_hp(), 1)
    env._prev_total_hp = max(env._total_party_hp() - 10, 0)

    _, comps = _call_reward(env)
    assert comps['r_heal'] > 0.0, (
        f"r_heal attendu > 0 après soin simulé, obtenu {comps['r_heal']}"
    )


def test_r_level_delta(env):
    """Forcer un delta de niveau : _prev_level_reward en-dessous du réel."""
    real = env._r_level()
    if real <= 0:
        pytest.skip("Pas de Pokémon en équipe dans ce save state")
    env._prev_level_reward = 0.0
    _, comps = _call_reward(env)
    assert comps['r_level'] == pytest.approx(real), (
        f"r_level attendu {real}, obtenu {comps['r_level']}"
    )


def test_reward_is_sum_of_components(env):
    """Le reward total doit être la somme exacte des 6 composantes."""
    reward, comps = _call_reward(env)
    expected = sum(comps.values())
    assert reward == pytest.approx(expected, abs=1e-6), (
        f"reward={reward} != sum(components)={expected}"
    )


def test_r_components_keys(env):
    """_r_components doit contenir exactement les 6 clés attendues."""
    _call_reward(env)
    assert set(env._r_components.keys()) == {
        'r_map', 'r_tile', 'r_heal', 'r_type', 'r_level', 'r_event'
    }
