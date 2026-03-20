"""
find_door.py — Charge un state et cherche les transitions de map
en marchant pas à pas dans toutes les directions depuis la position de spawn.

Usage:
    python src/utils/find_door.py --state states/06_pallet_town.state
"""

import argparse
from pyboy import PyBoy

ROM_PATH = 'ROMs/PokemonBlue.gb'

RAM_MAP_ID   = 0xD35E
RAM_PLAYER_X = 0xD362
RAM_PLAYER_Y = 0xD361

BUTTONS = ['up', 'down', 'left', 'right']

def press(pyboy, btn, ticks=24):
    pyboy.button_press(btn)
    for _ in range(ticks):
        pyboy.tick()
    pyboy.button_release(btn)
    for _ in range(ticks):
        pyboy.tick()

def load(pyboy, state_path):
    with open(state_path, 'rb') as f:
        pyboy.load_state(f)
    for _ in range(60):
        pyboy.tick()

def pos(pyboy):
    return pyboy.memory[RAM_MAP_ID], pyboy.memory[RAM_PLAYER_X], pyboy.memory[RAM_PLAYER_Y]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', required=True)
    parser.add_argument('--radius', type=int, default=8, help='Rayon de scan autour du spawn')
    args = parser.parse_args()

    pyboy = PyBoy(ROM_PATH, window='null', sound=False)
    pyboy.set_emulation_speed(0)

    load(pyboy, args.state)
    start_map, start_x, start_y = pos(pyboy)
    print(f"Spawn : map=0x{start_map:02X}  x={start_x}  y={start_y}")
    print(f"Scan rayon={args.radius} — recherche transitions de map...\n")

    found = []

    for dx in range(-args.radius, args.radius + 1):
        for dy in range(-args.radius, args.radius + 1):
            target_x = start_x + dx
            target_y = start_y + dy
            if target_x < 0 or target_y < 0:
                continue

            # Recharger le state pour chaque essai
            load(pyboy, args.state)

            # Se déplacer vers (target_x, target_y) step par step
            for _ in range(abs(dx)):
                btn = 'right' if dx > 0 else 'left'
                press(pyboy, btn)
            for _ in range(abs(dy)):
                btn = 'down' if dy > 0 else 'up'
                press(pyboy, btn)

            # Faire un pas de plus dans chaque direction pour tester la transition
            for btn in BUTTONS:
                load(pyboy, args.state)
                for _ in range(abs(dx)):
                    press(pyboy, 'right' if dx > 0 else 'left')
                for _ in range(abs(dy)):
                    press(pyboy, 'down' if dy > 0 else 'up')
                press(pyboy, btn)
                new_map, new_x, new_y = pos(pyboy)
                if new_map != start_map:
                    entry = (target_x, target_y, btn, f"0x{new_map:02X}", new_x, new_y)
                    if entry not in found:
                        found.append(entry)
                        print(f"  TRANSITION : depuis ({target_x},{target_y}) direction={btn}"
                              f"  →  map=0x{new_map:02X}  spawn=({new_x},{new_y})")

    if not found:
        print("Aucune transition trouvée dans ce rayon. Augmente --radius.")

    pyboy.stop()

if __name__ == '__main__':
    main()
