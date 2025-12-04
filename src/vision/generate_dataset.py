import sys
import os
import random
import glob
from pyboy import PyBoy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- CONFIG ---
ROM_PATH = "PokemonBlue.gb"
STATE_FILES = glob.glob("*.state")
DATASET_PATH = os.path.join("data", "dataset", "raw")
IMG_PATH = os.path.join(DATASET_PATH, "images")
LABEL_PATH = os.path.join(DATASET_PATH, "labels")

os.makedirs(IMG_PATH, exist_ok=True)
os.makedirs(LABEL_PATH, exist_ok=True)

SCREEN_W, SCREEN_H = 160, 144
ADDR_X_POS, ADDR_Y_POS, ADDR_MAP_N = 0xD362, 0xD361, 0xD35E

# --- ATLAS (Portes et Panneaux) ---
KNOWN_DOORS = {
    0:  [(5, 5), (13, 5), (12, 11)], 
    1:  [(23, 25), (29, 19), (32, 7), (21, 15)], 
    2:  [(13, 25), (23, 17), (16, 17), (7, 9)], 
    33: [(6, 5)] 
}
KNOWN_SIGNS = {
    0: [(3, 5), (11, 5), (7, 9), (13, 13)], 
    1: [(13, 19), (19, 27)], 
    2: [(11, 17), (25, 23)], 
    12: [(9, 13), (11, 27)], 
    33: [(9, 11)] 
}

class BoundingBox:
    def __init__(self, x, y, w, h):
        self.x1, self.y1, self.x2, self.y2 = x, y, x+w, y+h
    def get_center(self):
        return (self.x1 + (self.x2 - self.x1)/2, self.y1 + (self.y2 - self.y1)/2)
    def to_yolo(self, cid):
        w, h = self.x2 - self.x1, self.y2 - self.y1
        cx, cy = self.x1 + w/2, self.y1 + h/2
        return f"{cid} {max(0, min(1, cx/SCREEN_W)):.6f} {max(0, min(1, cy/SCREEN_H)):.6f} {max(0, min(1, w/SCREEN_W)):.6f} {max(0, min(1, h/SCREEN_H)):.6f}"

def get_labels(pyboy):
    yolo_labels = []
    
    # 1. RÉCUPÉRATION DES SPRITES (Algorithme Clustering V3 - Le Robuste)
    OAM_BASE = 0xFE00
    raw_tiles = []
    for i in range(40):
        addr = OAM_BASE + (i*4)
        sy, sx = pyboy.memory[addr]-16, pyboy.memory[addr+1]-8
        tile_id = pyboy.memory[addr+2]
        if -8 < sx < SCREEN_W and -8 < sy < SCREEN_H and tile_id < 0xF0:
            raw_tiles.append((sx, sy))

    entities = []
    while raw_tiles:
        tx, ty = raw_tiles.pop(0)
        cluster_x, cluster_y = [tx], [ty]
        for i in range(len(raw_tiles)-1, -1, -1):
            nx, ny = raw_tiles[i]
            if abs(nx - tx) < 12 and abs(ny - ty) < 12:
                cluster_x.append(nx); cluster_y.append(ny)
                raw_tiles.pop(i)
        entities.append(BoundingBox(min(cluster_x), min(cluster_y), 16, 16))

    # 2. IDENTIFICATION DU JOUEUR (Heuristique : Le plus proche du centre)
    # Même au bord de la map, le joueur reste l'élément le plus central par rapport à l'action
    player_ent = None
    min_dist = 999
    
    for e in entities:
        cx, cy = e.get_center()
        dist = ((cx - 80)**2 + (cy - 72)**2)**0.5
        if dist < min_dist: 
            min_dist = dist
            player_ent = e
    
    # Si on ne trouve pas de joueur "crédible" (trop loin du centre > 70px), on abandonne pour cette frame
    # (Évite de mapper des portes n'importe comment si le joueur clignote)
    if player_ent is None or min_dist > 70:
        return []

    # On enregistre les Sprites
    for e in entities:
        is_player = (e == player_ent)
        yolo_labels.append(e.to_yolo(0 if is_player else 1))

    # 3. CALIBRATION DYNAMIQUE DE LA CAMÉRA
    # On utilise la position RÉELLE du sprite joueur à l'écran pour caler le reste
    player_screen_x = player_ent.x1
    player_screen_y = player_ent.y1
    
    player_world_x = pyboy.memory[ADDR_X_POS] * 16
    player_world_y = pyboy.memory[ADDR_Y_POS] * 16
    
    # Le "Décalage Magique" : C'est la position 0,0 du monde par rapport à l'écran
    offset_x = player_screen_x - player_world_x
    offset_y = player_screen_y - player_world_y

    # 4. PLACEMENT DES OBJETS STATIQUES (PORTES / SIGNES)
    map_id = pyboy.memory[ADDR_MAP_N]
    
    def add_static(collection, cid):
        if map_id in collection:
            for wx, wy in collection[map_id]:
                # Position Monde (Pixels)
                world_x = wx * 16
                world_y = wy * 16
                
                # Position Écran = Monde + Décalage
                sx = world_x + offset_x
                sy = world_y + offset_y
                
                # Ajustement fin (souvent nécessaire car les sprites sont décalés de quelques pixels)
                # On centre la boîte sur la tuile
                # sy -= 4 
                
                if -16 < sx < SCREEN_W and -16 < sy < SCREEN_H:
                    yolo_labels.append(BoundingBox(sx, sy, 16, 16).to_yolo(cid))

    add_static(KNOWN_DOORS, 2)
    add_static(KNOWN_SIGNS, 3)

    return yolo_labels

def main():
    if not STATE_FILES: return
    # Nettoyage
    for f in glob.glob(os.path.join(IMG_PATH, "*")): os.remove(f)
    for f in glob.glob(os.path.join(LABEL_PATH, "*")): os.remove(f)

    print(f"🌍 Génération V7 (Calibration Dynamique) - 5000 images...")
    pyboy = PyBoy(ROM_PATH, window="SDL2", sound=False)
    pyboy.set_emulation_speed(0) 
    
    img_count = 0
    steps = 0
    TARGET = 5000
    
    while img_count < TARGET and pyboy.tick():
        if steps <= 0:
            try:
                state = random.choice(STATE_FILES)
                with open(state, "rb") as f: pyboy.load_state(f)
                for _ in range(10): pyboy.tick()
                steps = 100
                print(f"📂 {state}                         ", end="\r")
            except: 
                if state in STATE_FILES: STATE_FILES.remove(state)
                continue

        action = random.choice(['up', 'down', 'left', 'right'])
        pyboy.button(action)
        for _ in range(5): pyboy.tick()

        try:
            labels = get_labels(pyboy)
            if len(labels) > 0:
                name = f"train_{img_count:05d}"
                img = pyboy.screen.image
                if img.mode == 'RGBA': img = img.convert('RGB')
                img.save(os.path.join(IMG_PATH, name+".jpg"))
                with open(os.path.join(LABEL_PATH, name+".txt"), "w") as f:
                    f.write("\n".join(labels))
                img_count += 1
                steps -= 1
                if img_count % 50 == 0: print(f"📸 {img_count}/{TARGET}", end="\r")
        except: pass

    pyboy.stop()
    print("\n✅ Terminé !")

if __name__ == "__main__":
    main()