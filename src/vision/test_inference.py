import sys
import os
import cv2
import glob
import numpy as np
from pyboy import PyBoy
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- CONFIGURATION ---
# ⚠️ IMPORTANT : Vérifie que le nom du dossier correspond à ton dernier entraînement !
# Ça peut être "yolo_pokemon", "yolo_pokemon2", etc.
# Va voir dans ton dossier 'models/' pour être sûr.
MODEL_PATH = "models/yolo_pokemon_doors/weights/best.pt" 
ROM_PATH = "PokemonBlue.gb"

def main():
    # Récupérer la state en argument (ex: python test_inference.py jadielle.state)
    state_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # 1. Vérification du modèle
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Erreur : Modèle introuvable ici -> {MODEL_PATH}")
        print("vérifie le dossier 'models/' et mets à jour la ligne MODEL_PATH dans le script.")
        return

    print(f"🧠 Chargement du cerveau : {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Erreur chargement YOLO: {e}")
        return

    print("🎮 Lancement du jeu...")
    pyboy = PyBoy(ROM_PATH, window="SDL2", sound=False)
    pyboy.set_emulation_speed(1) # Vitesse normale pour que tu puisses jouer

    # Chargement de la sauvegarde .state
    if state_path and os.path.exists(state_path):
        with open(state_path, "rb") as f:
            pyboy.load_state(f)
        print(f"📂 State chargée : {state_path}")
        for _ in range(10): pyboy.tick()
    elif state_path:
        print(f"⚠️ State introuvable : {state_path}")

    print("✅ C'est parti ! (Appuie sur 'q' sur la fenêtre vidéo pour quitter)")

    while pyboy.tick():
        # --- A. CAPTURE ---
        screen = pyboy.screen.image
        if screen.mode == 'RGBA': screen = screen.convert('RGB')
        
        # Conversion Pillow -> OpenCV (BGR)
        frame = np.array(screen)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # --- B. INFÉRENCE (Vision) ---
        # conf=0.5 : On n'affiche que si l'IA est sûre à 50% minimum
        results = model(frame, imgsz=320, conf=0.5, verbose=False)

        # --- C. DESSIN ---
        for result in results:
            for box in result.boxes:
                # Coordonnées
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Couleurs & Labels (Format BGR pour OpenCV)
                label = "Unknown"
                color = (255, 255, 255) # Blanc par défaut
                
                if cls == 0:   # PLAYER
                    label = "Player"
                    color = (255, 0, 0)   # Bleu
                elif cls == 1: # NPC
                    label = "NPC"
                    color = (0, 0, 255)   # Rouge
                elif cls == 2: # DOOR
                    label = "Door"
                    color = (0, 255, 255) # Jaune
                elif cls == 3: # SIGN (Nouveau !)
                    label = "Sign"
                    color = (255, 0, 255) # Magenta/Violet

                # Dessiner le rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Dessiner le texte (avec un petit fond noir pour la lisibilité)
                text = f"{label} {conf:.0%}"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (x1, y1 - 15), (x1 + w, y1), color, -1) # Fond coloré
                cv2.putText(frame, text, (x1, y1 - 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # --- D. AFFICHAGE ---
        # On agrandit l'image x4 pour bien voir sur ton écran PC
        big_frame = cv2.resize(frame, (640, 576), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("YOLO Vision v1", big_frame)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pyboy.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
