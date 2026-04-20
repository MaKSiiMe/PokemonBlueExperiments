"""
test_env_step.py — Smoke test : 100 steps aléatoires.

Vérifie que :
  - obs respecte observation_space après chaque step
  - reward est un float
  - terminated et truncated sont des bool
  - info contient les clés attendues
"""

import numpy as np
import pytest


INFO_KEYS = {
    'map_id', 'player_x', 'player_y', 'steps_stuck',
    'unique_maps', 'max_level', 'n_badges', 'pokedex_owned', 'episode_steps',
    'ms_viridian', 'ms_forest', 'ms_pewter', 'ms_badge1', 'ms_mt_moon',
}
REWARD_KEYS = {'r_map', 'r_tile', 'r_event'}


def test_obs_at_reset(env):
    obs, info = env.reset()
    assert env.observation_space.contains(obs), "obs après reset hors observation_space"


def test_100_random_steps(env):
    env.reset()
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert env.observation_space.contains(obs), f"step {step}: obs hors observation_space"
        assert isinstance(reward, float), f"step {step}: reward n'est pas un float ({type(reward)})"
        assert isinstance(terminated, bool), f"step {step}: terminated n'est pas un bool"
        assert isinstance(truncated, bool), f"step {step}: truncated n'est pas un bool"
        assert INFO_KEYS.issubset(info.keys()), (
            f"step {step}: clés info manquantes : {INFO_KEYS - info.keys()}"
        )
        assert REWARD_KEYS.issubset(info.keys()), (
            f"step {step}: composantes reward manquantes : {REWARD_KEYS - info.keys()}"
        )
        assert np.isfinite(reward), f"step {step}: reward non fini ({reward})"

        if terminated or truncated:
            env.reset()


def test_obs_shapes(env):
    obs, _ = env.reset()
    assert obs['screen'].shape       == (3, 72, 80),  f"screen shape: {obs['screen'].shape}"
    assert obs['visited_mask'].shape == (1, 48, 48),  f"mask shape: {obs['visited_mask'].shape}"
    assert obs['ram'].shape          == (16,),         f"ram shape: {obs['ram'].shape}"


def test_obs_ranges(env):
    obs, _ = env.reset()
    assert obs['screen'].min() >= 0.0 and obs['screen'].max() <= 1.0, "screen hors [0,1]"
    assert obs['visited_mask'].min() >= 0.0 and obs['visited_mask'].max() <= 1.0
    assert obs['ram'].min() >= 0.0 and obs['ram'].max() <= 1.0, "ram hors [0,1]"
