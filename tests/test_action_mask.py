"""
test_action_mask.py — Vérifie la cohérence du masque d'action selon l'état du jeu.

Actions layout : ['up', 'down', 'left', 'right', 'a', 'b', 'start']
                   0      1       2       3        4    5      6
"""

import pytest
from src.emulator.ram_map import RAM_BATTLE, RAM_FADING


def _set_ram(env, addr, value):
    env.pyboy.memory[addr] = value


def test_overworld_all_actions_allowed(env):
    _set_ram(env, RAM_BATTLE, 0)
    _set_ram(env, RAM_FADING, 0)
    mask = env.action_masks()
    assert mask.all(), "Overworld : toutes les actions doivent être autorisées"


def test_fading_movement_disabled(env):
    _set_ram(env, RAM_BATTLE, 0)
    _set_ram(env, RAM_FADING, 1)
    mask = env.action_masks()
    assert not mask[0], "fading: up doit être masqué"
    assert not mask[1], "fading: down doit être masqué"
    assert not mask[2], "fading: left doit être masqué"
    assert not mask[3], "fading: right doit être masqué"
    assert not mask[6], "fading: start doit être masqué"
    assert mask[4], "fading: A doit rester autorisé"
    assert mask[5], "fading: B doit rester autorisé"
    _set_ram(env, RAM_FADING, 0)


def test_battle_movement_disabled(env):
    _set_ram(env, RAM_BATTLE, 1)
    mask = env.action_masks()
    assert not mask[0], "battle: up masqué"
    assert not mask[1], "battle: down masqué"
    assert not mask[2], "battle: left masqué"
    assert not mask[3], "battle: right masqué"
    assert not mask[5], "battle: B masqué"
    assert not mask[6], "battle: start masqué"
    _set_ram(env, RAM_BATTLE, 0)


def test_mask_is_7_bools(env):
    mask = env.action_masks()
    assert mask.shape == (7,), f"mask shape attendu (7,), obtenu {mask.shape}"
    assert mask.dtype.kind == 'b', "mask doit être de type bool"


def test_battle_always_has_one_action(env):
    _set_ram(env, RAM_BATTLE, 1)
    mask = env.action_masks()
    assert mask.any(), "battle: au moins une action doit toujours être disponible"
    _set_ram(env, RAM_BATTLE, 0)
