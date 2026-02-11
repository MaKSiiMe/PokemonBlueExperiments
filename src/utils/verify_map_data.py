import sys
import os
import glob
import cv2
import numpy as np
from pyboy import PyBoy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

ROM_PATH = "PokemonBlue.gb"
STATE_DIR = "states"

# Adresses mémoire
MEM_MAP_ID = 0xD35E
MEM_PLAYER_X = 0xD362
MEM_PLAYER_Y = 0xD361
MEM_WARP_COUNT = 0xD3AE
MEM_WARP_DATA = 0xD3AF
MEM_SIGN_COUNT = 0xD4B0
MEM_SIGN_DATA = 0xD4B1

# Position du joueur à l'écran
# Écran = 160x144 = 10x9 tiles
# Le joueur est sur la 5ème colonne (index 4) et 5ème ligne (index 4)
PLAYER_SCREEN_X = 64  # 4 * 16
PLAYER_SCREEN_Y = 64  # 4 * 16
TILE_SIZE = 16

def get_data(pyboy):
    """Récupère toutes les données de la map actuelle."""
    map_id = pyboy.memory[MEM_MAP_ID]
    px = pyboy.memory[MEM_PLAYER_X]
    py = pyboy.memory[MEM_PLAYER_Y]
    
    doors = []
    num_warps = pyboy.memory[MEM_WARP_COUNT]
    if 0 < num_warps < 20:
        for i in range(num_warps):
            addr = MEM_WARP_DATA + (i * 4)
            y = pyboy.memory[addr]
            x = pyboy.memory[addr + 1]
            dest_map = pyboy.memory[addr + 3]
            if 0 < x < 100 and 0 < y < 100:
                doors.append((x, y, dest_map))
    
    signs = []
    num_signs = pyboy.memory[MEM_SIGN_COUNT]
    if 0 < num_signs < 20:
        for i in range(num_signs):
            addr = MEM_SIGN_DATA + (i * 3)
            y = pyboy.memory[addr]
            x = pyboy.memory[addr + 1]
            if 0 < x < 100 and 0 < y < 100:
                signs.append((x, y))
    
    return map_id, px, py, doors, signs

def world_to_screen(obj_x, obj_y, player_x, player_y):
    """Convertit coordonnées monde en coordonnées écran."""
    screen_x = PLAYER_SCREEN_X + (obj_x - player_x) * TILE_SIZE
    screen_y = PLAYER_SCREEN_Y + (obj_y - player_y) * TILE_SIZE
    return int(screen_x), int(screen_y)

def draw_overlay(frame, map_id, px, py, doors, signs):
    """Dessine les portes et panneaux sur la frame."""
    img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (480, 432), interpolation=cv2.INTER_NEAREST)
    scale = 3  # 480/160
    
    # Dessiner les PORTES (cyan)
    for door in doors:
        x, y = door[0], door[1]
        sx, sy = world_to_screen(x, y, px, py)
        sx, sy = sx * scale, sy * scale
        w, h = TILE_SIZE * scale, TILE_SIZE * scale
        if -w < sx < 480 and -h < sy < 432:
            cv2.rectangle(img, (sx, sy), (sx + w, sy + h), (255, 255, 0), 2)
            cv2.putText(img, "D", (sx + 4, sy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Dessiner les SIGNS (vert)
    for sign in signs:
        x, y = sign[0], sign[1]
        sx, sy = world_to_screen(x, y, px, py)
        sx, sy = sx * scale, sy * scale
        w, h = TILE_SIZE * scale, TILE_SIZE * scale
        if -w < sx < 480 and -h < sy < 432:
            cv2.rectangle(img, (sx, sy), (sx + w, sy + h), (0, 255, 0), 2)
            cv2.putText(img, "S", (sx + 4, sy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Marquer le joueur (rouge)
    px_screen = PLAYER_SCREEN_X * scale
    py_screen = PLAYER_SCREEN_Y * scale
    cv2.rectangle(img, (px_screen, py_screen), (px_screen + TILE_SIZE*scale, py_screen + TILE_SIZE*scale), (0, 0, 255), 2)
    cv2.putText(img, "P", (px_screen + 4, py_screen + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Info
    cv2.putText(img, f"Map:{map_id} Pos:({px},{py}) Doors:{len(doors)} Signs:{len(signs)}", 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

def main():
    state_files = sorted(glob.glob(os.path.join(STATE_DIR, "*.state")))
    
    if not state_files:
        print("❌ Aucun fichier .state trouvé")
        return
    
    pyboy = PyBoy(ROM_PATH, window="null", sound=False)
    pyboy.set_emulation_speed(0)
    
    idx = 0
    print("🎮 Contrôles: [ESPACE] Suivant | [B] Précédent | [Q] Quitter")
    print("   Les rectangles CYAN = Portes (D), VERT = Panneaux (S), ROUGE = Joueur")
    
    while True:
        state = state_files[idx]
        
        with open(state, "rb") as f:
            pyboy.load_state(f)
        
        for _ in range(10):
            pyboy.tick()
        
        map_id, px, py, doors, signs = get_data(pyboy)
        frame = pyboy.screen.image
        img = draw_overlay(frame, map_id, px, py, doors, signs)
        
        cv2.putText(img, f"[{idx+1}/{len(state_files)}] {os.path.basename(state)}", 
                    (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Verification - Portes & Panneaux", img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') or key == ord('n'):
            idx = (idx + 1) % len(state_files)
        elif key == ord('b') or key == ord('p'):
            idx = (idx - 1) % len(state_files)
    
    pyboy.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()