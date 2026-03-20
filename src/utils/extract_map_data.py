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

ROM_PATH  = 'ROMs/PokemonBlue.gb'
STATE_DIR = 'states'

# RAM addresses (voir src/emulator/ram_map.py)
RAM_MAP_ID    = 0xD35E
RAM_PLAYER_X  = 0xD362
RAM_PLAYER_Y  = 0xD361
RAM_WARP_COUNT = 0xD3AE
RAM_WARP_DATA  = 0xD3AF
RAM_SIGN_COUNT = 0xD4B0
RAM_SIGN_DATA  = 0xD4B1


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
