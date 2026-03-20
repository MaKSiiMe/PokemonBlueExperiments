"""
dump_states.py — Charge chaque save state et affiche map_id / x / y de spawn.
Utile pour construire le curriculum de waypoints.

Usage:
    python src/utils/dump_states.py
"""

import os
import glob
from pyboy import PyBoy

ROM_PATH   = 'ROMs/PokemonBlue.gb'
STATES_DIR = 'states/'

RAM_MAP_ID   = 0xD35E
RAM_PLAYER_X = 0xD362
RAM_PLAYER_Y = 0xD361


def dump_state(pyboy: PyBoy, state_path: str) -> tuple[int, int, int]:
    with open(state_path, 'rb') as f:
        pyboy.load_state(f)
    for _ in range(60):
        pyboy.tick()
    return (
        pyboy.memory[RAM_MAP_ID],
        pyboy.memory[RAM_PLAYER_X],
        pyboy.memory[RAM_PLAYER_Y],
    )


def main():
    if not os.path.exists(ROM_PATH):
        print(f"ROM introuvable : {ROM_PATH}")
        return

    pyboy = PyBoy(ROM_PATH, window='null', sound=False)
    pyboy.set_emulation_speed(0)

    states = sorted(glob.glob(os.path.join(STATES_DIR, '*.state')))
    if not states:
        print(f"Aucun state trouvé dans {STATES_DIR}")
        pyboy.stop()
        return

    print(f"{'State':<45} {'map_id':>8}  {'x':>4}  {'y':>4}")
    print('-' * 65)

    for state_path in states:
        name = os.path.basename(state_path)
        try:
            map_id, x, y = dump_state(pyboy, state_path)
            print(f"{name:<45} 0x{map_id:02X} ({map_id:3d})  {x:4d}  {y:4d}")
        except Exception as e:
            print(f"{name:<45} ERROR: {e}")

    pyboy.stop()


if __name__ == '__main__':
    main()
