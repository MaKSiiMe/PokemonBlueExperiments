import sys
import os
import glob
from pyboy import PyBoy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- CONFIGURATION ---
ROM_PATH = "PokemonBlue.gb"

STATE_DIR = "states"
STATE_FILES = glob.glob(os.path.join(STATE_DIR, "*.state"))

# --- ADRESSES MÉMOIRE (US VERSION) ---
MEM_MAP_ID = 0xD35E
MEM_PLAYER_X = 0xD362
MEM_PLAYER_Y = 0xD361

# WARPS (Portes)
MEM_WARP_COUNT = 0xD3AE
MEM_WARP_DATA  = 0xD3AF

# SIGNS
MEM_SIGN_COUNT = 0xD4B0
MEM_SIGN_DATA  = 0xD4B1

def scan_state(pyboy, filename):
    with open(filename, "rb") as f:
        pyboy.load_state(f)
    
    # Ticks pour charger la RAM
    for _ in range(10):
        pyboy.tick()

    map_id = pyboy.memory[MEM_MAP_ID]
    player_x = pyboy.memory[MEM_PLAYER_X]
    player_y = pyboy.memory[MEM_PLAYER_Y]
    
    print(f"📂 {os.path.basename(filename):40} | Map: {map_id:3} | Pos: ({player_x:2}, {player_y:2})", end="")

    # WARPS
    doors = []
    num_warps = pyboy.memory[MEM_WARP_COUNT]
    if 0 < num_warps < 20:
        for i in range(num_warps):
            addr = MEM_WARP_DATA + (i * 4)
            y = pyboy.memory[addr]
            x = pyboy.memory[addr + 1]
            # Filtrer les coordonnées invalides
            if 0 < x < 100 and 0 < y < 100:
                doors.append((x, y))
    
    # SIGNS
    signs = []
    num_signs = pyboy.memory[MEM_SIGN_COUNT]
    if 0 < num_signs < 20:
        for i in range(num_signs):
            addr = MEM_SIGN_DATA + (i * 3)
            y = pyboy.memory[addr]
            x = pyboy.memory[addr + 1]
            # Filtrer les coordonnées invalides
            if 0 < x < 100 and 0 < y < 100:
                signs.append((x, y))
    
    print(f" | 🚪 {len(doors)} | 📜 {len(signs)}")

    return map_id, doors, signs

def main():
    if not STATE_FILES:
        print(f"❌ Aucun fichier .state trouvé dans '{STATE_DIR}/'")
        return

    print(f"🕵️  SCANNER DE MAP ({len(STATE_FILES)} fichiers)...\n")
    
    pyboy = PyBoy(ROM_PATH, window="null", sound=False)
    pyboy.set_emulation_speed(0)

    final_doors = {}
    final_signs = {}

    # TOUS les fichiers maintenant !
    for state in sorted(STATE_FILES):
        try:
            mid, d, s = scan_state(pyboy, state)
            
            if mid not in final_doors: final_doors[mid] = set()
            if mid not in final_signs: final_signs[mid] = set()
            
            for door in d: final_doors[mid].add(door)
            for sign in s: final_signs[mid].add(sign)
            
        except Exception as e:
            print(f"⚠️ Erreur sur {state}: {e}")

    pyboy.stop()
    
    print("\n" + "="*60)
    print("✅ RÉSULTAT À COPIER DANS generate_dataset.py")
    print("="*60)
    
    print("\nKNOWN_DOORS = {")
    for mid, coords in sorted(final_doors.items()):
        if coords:
            sorted_coords = sorted(list(coords))
            print(f"    {mid}: {sorted_coords},")
    print("}")

    print("\nKNOWN_SIGNS = {")
    for mid, coords in sorted(final_signs.items()):
        if coords:
            sorted_coords = sorted(list(coords))
            print(f"    {mid}: {sorted_coords},")
    print("}")
    
    # Stats
    total_doors = sum(len(v) for v in final_doors.values())
    total_signs = sum(len(v) for v in final_signs.values())
    print(f"\n📊 Total: {total_doors} portes, {total_signs} panneaux sur {len(final_doors)} maps")

if __name__ == "__main__":
    main()
