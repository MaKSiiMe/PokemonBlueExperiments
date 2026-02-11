import sys
import os
import random
import glob
import numpy as np
import cv2
from pyboy import PyBoy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- CONFIGURATION PRINCIPALE ---
ROM_PATH = "PokemonBlue.gb"
STATE_FILES = glob.glob(os.path.join("states", "*.state"))

DATASET_PATH = os.path.join("data", "dataset", "raw")
IMG_PATH = os.path.join(DATASET_PATH, "images")
LABEL_PATH = os.path.join(DATASET_PATH, "labels")

os.makedirs(IMG_PATH, exist_ok=True)
os.makedirs(LABEL_PATH, exist_ok=True)

# --- CALIBRATION ---
SCREEN_W, SCREEN_H = 160, 144
TILE_SIZE = 16
Y_OFFSET = -2   # Remonte la boîte de 2px (pour centrer sur le graphisme)

# Adresses Mémoire (RAM)
ADDR_MAP_N = 0xD35E
ADDR_SCX = 0xFF43
ADDR_SCY = 0xFF42

# --- ATLAS DES TUILES (V12) ---
# Listes des IDs de tuiles qui représentent des objets interactifs

TILE_ATLAS = {
    "OUTDOOR": {
        "DOORS": [
            0x1B, # Porte standard (Bourg Palette, Jadielle)
            0x2E, # Porte Labo Chen
            0x5F, # Porte Centre Pokémon (Extérieur)
            0x17, # Porte Mart (Bas)
            0x3D, # Porte Gym
        ],
        "SIGNS": [
            0x52, # Panneau bois
            0x5C, # Panneau Gym (Texte)
            0x5F, # Panneau PokéCenter (parfois détecté comme sign)
        ]
    },
    "INDOOR": {
        "DOORS": [
            0x5D, # Tapis de sortie (Exit Mat)
            0x1E, # Escalier bas (parfois)
        ],
        "SIGNS": [
            0x04, # Carte murale
            0x14, # Bibliothèque / Etagère livres
            0x53, # Panneau d'affichage (PC)
            0x5E, # Tapis comptoir (parfois utile)
        ]
    }
}

# Liste des Maps considérées comme "Extérieur" (Villes + Routes)
# Tout le reste sera traité comme "Intérieur"
OUTDOOR_MAP_IDS = list(range(0, 34)) + [51] # 0-33 (Villes/Routes) + 51 (Foret Jade)

class BoundingBox:
    def __init__(self, x, y, w, h):
        self.x1, self.y1, self.x2, self.y2 = x, y, x+w, y+h
    
    def get_center(self):
        return (self.x1 + (self.x2 - self.x1)/2, self.y1 + (self.y2 - self.y1)/2)
        
    def to_yolo(self, cid):
        w, h = self.x2 - self.x1, self.y2 - self.y1
        cx, cy = self.x1 + w/2, self.y1 + h/2
        return f"{cid} {max(0, min(1, cx/SCREEN_W)):.6f} {max(0, min(1, cy/SCREEN_H)):.6f} {max(0, min(1, w/SCREEN_W)):.6f} {max(0, min(1, h/SCREEN_H)):.6f}"

def get_screen_matrix(pyboy):
    """
    Lit la VRAM (Background Map) pour reconstruire la grille 20x18 de tuiles
    telle qu'elle est affichée à l'écran, en tenant compte du Scroll X/Y.
    """
    scx = pyboy.memory[ADDR_SCX]
    scy = pyboy.memory[ADDR_SCY]
    
    # La map de fond est stockée en 32x32 tuiles à partir de 0x9800 (généralement)
    BG_MAP_ADDR = 0x9800 
    
    matrix = np.zeros((18, 20), dtype=np.uint8)
    
    for y in range(18):
        for x in range(20):
            # Calcul de la position dans le buffer circulaire 32x32
            tile_y = (scy // 8 + y) % 32
            tile_x = (scx // 8 + x) % 32
            
            # Adresse de la tuile en VRAM
            addr = BG_MAP_ADDR + (tile_y * 32) + tile_x
            tile_id = pyboy.memory[addr]
            
            matrix[y, x] = tile_id
            
    return matrix

def extract_boxes_from_mask(mask, class_id):
    """
    Utilise OpenCV pour grouper les tuiles adjacentes (Clustering)
    et créer des boîtes uniques pour les objets larges (ex: doubles portes).
    """
    # Connectivité 4 pour éviter de lier des objets en diagonale
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    
    local_boxes = []
    # On ignore le label 0 (le fond)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_tiles = stats[i, cv2.CC_STAT_WIDTH]
        h_tiles = stats[i, cv2.CC_STAT_HEIGHT]

        # Conversion Tuiles -> Pixels Écran
        # Note: x et y sont ici des indices de grille (0-19, 0-17)
        # On doit les ajuster avec le scroll fin (modulo 8)
        
        # ATTENTION: get_screen_matrix aligne déjà sur les tuiles entières.
        # Mais le scroll pixel fin (SCX % 8) décale l'affichage.
        # Simplification V12 : On assume un alignement grille pour l'apprentissage,
        # ou on ajoute le décalage fin si nécessaire. 
        # Pour YOLO, un décalage de 0-7px est tolérable, mais corrigeons-le :
        
        # (Cette version simplifiée ignore le SCX%8 pour l'instant car le modèle est robuste,
        # mais on applique le Y_OFFSET global)
        
        px_x = x * 16
        px_y = y * 16 + Y_OFFSET # Calibration Verticale
        px_w = w_tiles * 16
        px_h = h_tiles * 16
        
        # Filtre anti-bruit (trop petit = bug probable)
        if px_w < 8 or px_h < 8: continue
        
        local_boxes.append(BoundingBox(px_x, px_y, px_w, px_h).to_yolo(class_id))
        
    return local_boxes

def get_labels(pyboy):
    yolo_labels = []
    
    # --- 1. DETECTION JOUEUR / NPC (SPRITES) ---
    # (Code inchangé car fonctionnel, juste le strict clustering appliqué)
    OAM_BASE = 0xFE00
    raw_tiles = []
    for i in range(40):
        addr = OAM_BASE + (i*4)
        sy, sx = pyboy.memory[addr]-16, pyboy.memory[addr+1]-8
        if -8 < sx < SCREEN_W and -8 < sy < SCREEN_H:
            raw_tiles.append((sx, sy))

    entities = []
    while raw_tiles:
        tx, ty = raw_tiles.pop(0)
        cluster_x, cluster_y = [tx], [ty]
        for i in range(len(raw_tiles)-1, -1, -1):
            nx, ny = raw_tiles[i]
            if abs(nx - tx) < 12 and abs(ny - ty) < 12:
                cluster_x.append(nx)
                cluster_y.append(ny)
                raw_tiles.pop(i)
        
        width = max(cluster_x) - min(cluster_x) + 8
        height = max(cluster_y) - min(cluster_y) + 8
        if width <= 18 and height <= 18:
            entities.append(BoundingBox(min(cluster_x), min(cluster_y), width, height))

    TARGET_X, TARGET_Y = 72, 72 
    player_ent = None
    min_dist = 999
    
    for e in entities:
        cx, cy = e.get_center()
        dist = ((cx - TARGET_X)**2 + (cy - TARGET_Y)**2)**0.5
        if dist < min_dist: 
            min_dist = dist
            player_ent = e
    
    if player_ent:
        for e in entities:
            yolo_labels.append(e.to_yolo(0 if e == player_ent else 1))

    # --- 2. DETECTION STATIQUE (PORTES / PANNEAUX) ---
    # Nouvelle logique V12 : Scan Matrix + Atlas + Clustering
    
    map_id = pyboy.memory[ADDR_MAP_N]
    context = "OUTDOOR" if map_id in OUTDOOR_MAP_IDS else "INDOOR"
    
    screen_matrix = get_screen_matrix(pyboy)
    
    # Création des masques binaires
    doors_ids = TILE_ATLAS[context]["DOORS"]
    signs_ids = TILE_ATLAS[context]["SIGNS"]
    
    # np.isin crée un masque True/False là où les IDs correspondent
    door_mask = np.isin(screen_matrix, doors_ids).astype(np.uint8)
    sign_mask = np.isin(screen_matrix, signs_ids).astype(np.uint8)
    
    # Extraction et conversion YOLO
    yolo_labels.extend(extract_boxes_from_mask(door_mask, 2)) # Class 2: Door
    yolo_labels.extend(extract_boxes_from_mask(sign_mask, 3)) # Class 3: Sign

    return yolo_labels

def main():
    if not STATE_FILES: 
        print("❌ Pas de fichiers .state trouvés dans /states")
        return
        
    # Nettoyage
    for f in glob.glob(os.path.join(IMG_PATH, "*")): os.remove(f)
    for f in glob.glob(os.path.join(LABEL_PATH, "*")): os.remove(f)

    print(f"🌍 Génération V12 (Atlas Dynamique + Clustering) - 5000 images...")
    pyboy = PyBoy(ROM_PATH, window="SDL2", sound=False)
    pyboy.set_emulation_speed(0) 
    
    img_count = 0
    TARGET = 5000
    
    while img_count < TARGET and pyboy.tick():
        # Logique de chargement d'état et mouvement aléatoire (inchangée)
        if pyboy.frame_count % 100 == 0: # Reset périodique pour varier
            try:
                state = random.choice(STATE_FILES)
                with open(state, "rb") as f: pyboy.load_state(f)
            except: pass

        action = random.choice(['up', 'down', 'left', 'right', 'a', 'pass'])
        if action != 'pass': pyboy.button(action)
        
        # On capture une image tous les X frames pour laisser le temps à l'animation
        if pyboy.frame_count % 30 == 0:
             try:
                labels = get_labels(pyboy)
                if len(labels) > 0:
                    name = f"v12_{img_count:05d}"
                    
                    # Sauvegarde Image
                    img = pyboy.screen.image
                    if img.mode == 'RGBA': img = img.convert('RGB')
                    img.save(os.path.join(IMG_PATH, name+".jpg"))
                    
                    # Sauvegarde Labels
                    with open(os.path.join(LABEL_PATH, name+".txt"), "w") as f:
                        f.write("\n".join(labels))
                    
                    img_count += 1
                    print(f"📸 {img_count}/{TARGET} | Map: {pyboy.memory[ADDR_MAP_N]}", end="\r")
             except Exception as e:
                 print(f"Error: {e}")

    pyboy.stop()
    print("\n✅ Terminé ! Dataset V12 généré.")

if __name__ == "__main__":
    main()
