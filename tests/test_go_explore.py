"""
test_go_explore.py — Vérifie capture_state() / reset_from_state().

Garantit que :
  - Un savestate capturé peut être rechargé (round-trip).
  - La position et la map sont identiques après rechargement.
  - L'observation retournée respecte l'observation_space.
  - Le step_count est remis à 0 après reset_from_state().
"""

import pytest
from src.emulator.ram_map import RAM_PLAYER_X, RAM_PLAYER_Y, RAM_MAP_ID


def test_capture_returns_bytes(env):
    state = env.capture_state()
    assert isinstance(state, bytes), "capture_state() doit retourner des bytes"
    assert len(state) > 0, "capture_state() ne doit pas retourner des bytes vides"


def test_round_trip_position(env):
    """Position (x, y, map_id) identique avant/après capture+restore."""
    x_before   = env._r(RAM_PLAYER_X)
    y_before   = env._r(RAM_PLAYER_Y)
    map_before = env._r(RAM_MAP_ID)

    state = env.capture_state()

    # Avancer quelques steps pour modifier la position
    for _ in range(10):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    obs, _ = env.reset_from_state(state)

    assert env._r(RAM_PLAYER_X) == x_before,   "X différent après restore"
    assert env._r(RAM_PLAYER_Y) == y_before,   "Y différent après restore"
    assert env._r(RAM_MAP_ID)   == map_before, "map_id différent après restore"


def test_reset_from_state_obs_valid(env):
    state = env.capture_state()
    obs, info = env.reset_from_state(state)
    assert env.observation_space.contains(obs), "obs après reset_from_state hors observation_space"
    assert isinstance(info, dict)


def test_reset_from_state_resets_step_count(env):
    state = env.capture_state()
    for _ in range(5):
        env.step(env.action_space.sample())
    assert env._step_count > 0

    env.reset_from_state(state)
    assert env._step_count == 0, "step_count doit être 0 après reset_from_state"


def test_reset_from_state_resets_steps_stuck(env):
    state = env.capture_state()
    env._steps_stuck = 99
    env.reset_from_state(state)
    assert env._steps_stuck == 0, "steps_stuck doit être 0 après reset_from_state"


def test_double_capture_independent(env):
    """Deux savestates capturés à des moments différents sont indépendants."""
    state_a = env.capture_state()
    for _ in range(5):
        env.step(env.action_space.sample())
    state_b = env.capture_state()

    assert state_a != state_b, "Deux captures à des moments différents doivent différer"

    # Restaurer A puis B : pas d'erreur
    env.reset_from_state(state_a)
    env.reset_from_state(state_b)
