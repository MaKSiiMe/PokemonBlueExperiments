"""Fixtures partagées.

Les tests qui ont besoin de la ROM ou d'un savestate sont sautés s'ils sont absents
(ils ne sont jamais versionnés). Chemins surchargeables par variables d'environnement :
    POKEBLUE_ROM     (défaut : ROMs/PokemonBlue.gb)
    POKEBLUE_STATES  (défaut : states/)
"""

import os
from pathlib import Path

import pytest

ROM_PATH   = Path(os.environ.get('POKEBLUE_ROM', 'ROMs/PokemonBlue.gb'))
STATES_DIR = Path(os.environ.get('POKEBLUE_STATES', 'states'))
INIT_STATE = STATES_DIR / '00_pallet_town.state'


@pytest.fixture(scope='session')
def rom_path():
    if not ROM_PATH.exists():
        pytest.skip(f"ROM introuvable : {ROM_PATH}")
    return str(ROM_PATH)


@pytest.fixture
def state_path():
    """Retourne une fonction `nom -> chemin` qui saute le test si le savestate manque."""
    def _get(name: str) -> str:
        path = STATES_DIR / name
        if not path.exists():
            pytest.skip(f"savestate introuvable : {path}")
        return str(path)
    return _get


@pytest.fixture
def env(rom_path):
    from src.emulator.pokemon_env import PokemonBlueEnv
    e = PokemonBlueEnv(rom_path=rom_path, init_state=str(INIT_STATE), headless=True, speed=0)
    e.reset()
    yield e
    e.close()
