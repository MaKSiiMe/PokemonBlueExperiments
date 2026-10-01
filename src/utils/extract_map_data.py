"""
extract_map_data.py — Scanne les warps et panneaux dans chaque save state.

Lit les adresses RAM après chargement pour collecter les portes/transitions
et panneaux de chaque map. Utile pour construire KNOWN_DOORS / KNOWN_SIGNS.

Usage :
    python src/utils/extract_map_data.py
"""

import os
import glob
from pyboy import PyBoy
from pokeblue.state import ram_symbols as sym

ROM_PATH  = 'ROMs/PokemonBlue.gb'
STATE_DIR = 'states'

# RAM addresses (voir src/emulator/ram_map.py)
RAM_MAP_ID    = sym.W_CUR_MAP
RAM_PLAYER_X  = sym.W_X_COORD
RAM_PLAYER_Y  = sym.W_Y_COORD
RAM_WARP_COUNT = sym.W_NUMBER_OF_WARPS
RAM_WARP_DATA  = sym.W_WARP_ENTRIES
RAM_SIGN_COUNT = sym.W_NUM_SIGNS
RAM_SIGN_DATA  = sym.W_SIGN_COORDS


def scan_state(pyboy: PyBoy, path: str):
    with open(path, 'rb') as f:
        pyboy.load_state(f)
    for _ in range(60):
        pyboy.tick()

    map_id   = pyboy.memory[RAM_MAP_ID]
    player_x = pyboy.memory[RAM_PLAYER_X]
    player_y = pyboy.memory[RAM_PLAYER_Y]

    print(f"{os.path.basename(path):<45} map={map_id:3}  pos=({player_x:2},{player_y:2})", end='')

    doors = []
    num_warps = pyboy.memory[RAM_WARP_COUNT]
    if 0 < num_warps < 20:
        for i in range(num_warps):
            addr = RAM_WARP_DATA + i * 4
            y, x = pyboy.memory[addr], pyboy.memory[addr + 1]
            if 0 < x < 100 and 0 < y < 100:
                doors.append((x, y))

    signs = []
    num_signs = pyboy.memory[RAM_SIGN_COUNT]
    if 0 < num_signs < 20:
        for i in range(num_signs):
            addr = RAM_SIGN_DATA + i * 3
            y, x = pyboy.memory[addr], pyboy.memory[addr + 1]
            if 0 < x < 100 and 0 < y < 100:
                signs.append((x, y))

    print(f"  warps={len(doors)}  signs={len(signs)}")
    return map_id, doors, signs


def main():
    state_files = sorted(glob.glob(os.path.join(STATE_DIR, '*.state')))
    if not state_files:
        print(f"Aucun .state trouvé dans {STATE_DIR}/")
        return

    print(f"Scan de {len(state_files)} states...\n")

    pyboy = PyBoy(ROM_PATH, window='null', sound=False)
    pyboy.set_emulation_speed(0)

    all_doors: dict[int, set] = {}
    all_signs: dict[int, set] = {}

    for path in state_files:
        try:
            mid, doors, signs = scan_state(pyboy, path)
            all_doors.setdefault(mid, set()).update(doors)
            all_signs.setdefault(mid, set()).update(signs)
        except Exception as e:
            print(f"  Erreur {os.path.basename(path)}: {e}")

    pyboy.stop()

    print("\n" + "=" * 50)
    print("KNOWN_DOORS = {")
    for mid, coords in sorted(all_doors.items()):
        if coords:
            print(f"    {mid}: {sorted(coords)},")
    print("}")

    print("\nKNOWN_SIGNS = {")
    for mid, coords in sorted(all_signs.items()):
        if coords:
            print(f"    {mid}: {sorted(coords)},")
    print("}")


if __name__ == '__main__':
    main()
