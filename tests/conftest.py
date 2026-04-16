"""Fixtures partagées pour les tests PokemonBlueEnv."""

import os
import pytest

ROM_PATH   = 'ROMs/PokemonBlue.gb'
INIT_STATE = 'states/00_pallet_town.state'


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tests longs (>10s)")


@pytest.fixture(scope='session')
def rom_path():
    if not os.path.exists(ROM_PATH):
        pytest.skip(f"ROM introuvable : {ROM_PATH}")
    return ROM_PATH


@pytest.fixture
def env(rom_path):
    from src.emulator.pokemon_env import PokemonBlueEnv
    e = PokemonBlueEnv(rom_path=rom_path, init_state=INIT_STATE, headless=True, speed=0)
    e.reset()
    yield e
    e.close()
