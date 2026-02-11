import sys
import os
import random
import glob
import cv2
import numpy as np
from pyboy import PyBoy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- ON IMPORTE TA LOGIQUE V8 ---
# Assure-toi que src/vision/generate_dataset.py est bien ta dernière version (V8)
from src.vision.generate_dataset import get_labels, ROM_PATH, SCREEN_W, SCREEN_H, KNOWN_DOORS

# Dossier des states
STATE_FILES = glob.glob(os.path.join("states", "*.state"))

# Couleurs pour le debug
COLORS = {
    0: (255, 0, 0),   # Player (Bleu)
    1: (0, 0, 255),   # NPC (Rouge)
    2: (0, 255, 255), # Door (Jaune)
    3: (255, 0, 255)  # Sign (Violet)
}

def draw_yolo_box_on_cv2(img, label_line):
    h, w, _ = img.shape
    parts = label_line.split()
    cls = int(parts[0])
    cx, cy, bw, bh = map(float, parts[1:])
    
    # Conversion YOLO -> Pixels
    x1 = int((cx - bw/2) * w)
    y1 = int((cy - bh/2) * h)
    x2 = int((cx + bw/2) * w)
    y2 = int((cy + bh/2) * h)
    
    color = COLORS.get(cls, (255, 255, 255))
    # Boîte épaisse (2px)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # Petit point au centre pour vérifier la précision
    cv2.circle(img, (int(cx*w), int(cy*h)), 2, color, -1)

def main():
    if not STATE_FILES:
        print("❌ Pas de states dans le dossier 'states/'")
        return

    print("🕵️  LIVE DEBUG GENERATOR (V8 Logic)")
    print("Appuie sur ESPACE pour changer de situation, Q pour quitter.")
    
    # Mode "null" car on utilise notre propre fenêtre OpenCV
    pyboy = PyBoy(ROM_PATH, window="null", sound=False)
    pyboy.set_emulation_speed(0)

    while True:
        # 1. Charger une situation au hasard
        state = random.choice(STATE_FILES)
        try:
            with open(state, "rb") as f: pyboy.load_state(f)
        except: continue
            
        for _ in range(10): pyboy.tick()

        # 2. Faire un petit mouvement aléatoire (pour tester la calibration)
        action = random.choice(['up', 'down', 'left', 'right', 'pass'])
        if action != 'pass':
            pyboy.button(action)
            for _ in range(5): pyboy.tick()

        # 3. RÉCUPÉRER L'IMAGE ET LES LABELS (Cœur du test)
        screen = pyboy.screen.image
        if screen.mode == 'RGBA': screen = screen.convert('RGB')
        frame_bgr = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
        
        # C'est cette fonction qu'on teste !
        labels = get_labels(pyboy)
        
        # 4. DESSINER LE RÉSULTAT
        obs_frame = frame_bgr.copy()
        doors_found = 0
        for label in labels:
            draw_yolo_box_on_cv2(obs_frame, label)
            if label.startswith("2 "): doors_found += 1

        # Informations de debug à l'écran
        map_id = pyboy.memory[0xD35E]
        px, py = pyboy.memory[0xD362], pyboy.memory[0xD361]
        
        info_text = f"Map:{map_id} Pos:({px},{py}) DoorsInDB:{len(KNOWN_DOORS.get(map_id, []))} Found:{doors_found}"
        
        # Agrandir pour bien voir
        big_frame = cv2.resize(obs_frame, (640, 576), interpolation=cv2.INTER_NEAREST)
        
        # Afficher le texte (Vert si OK, Rouge si problème potentiel)
        color_txt = (0, 255, 0) if doors_found > 0 or len(KNOWN_DOORS.get(map_id, [])) == 0 else (0, 0, 255)
        cv2.putText(big_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_txt, 2)
        cv2.putText(big_frame, os.path.basename(state), (10, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow("LIVE DEBUG (Space: Next, Q: Quit)", big_frame)
        
        # Attendre une touche
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    pyboy.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
