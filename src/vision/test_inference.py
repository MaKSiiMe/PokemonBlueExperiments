import sys
import os
import cv2
import numpy as np
from pyboy import PyBoy
from ultralytics import YOLO

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
)

MODEL_PATH = "models/yolo_pokemon_doors2/weights/best.pt"
ROM_PATH = "PokemonBlue.gb"
STATE_PATH = "states/init.state"


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        state_path = (arg if arg.startswith("states/") or "/" in arg
                      else f"states/{arg}")
    else:
        state_path = STATE_PATH

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Erreur : Modèle introuvable ici -> {MODEL_PATH}")
        print("vérifie le dossier 'models/' et mets à jour MODEL_PATH.")
        return

    print(f"🧠 Chargement du cerveau : {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Erreur chargement YOLO: {e}")
        return

    print("🎮 Lancement du jeu...")
    pyboy = PyBoy(ROM_PATH, window="SDL2", sound=False)
    pyboy.set_emulation_speed(1)

    if state_path and os.path.exists(state_path):
        with open(state_path, "rb") as f:
            pyboy.load_state(f)
        print(f"📂 State chargée : {state_path}")
        for _ in range(10):
            pyboy.tick()
    elif state_path:
        print(f"⚠️ State introuvable : {state_path}")
    else:
        print("⚠️ Aucune state spécifiée, démarrage normal...")

    print("✅ C'est parti ! (Appuie sur 'q' pour quitter)")

    while pyboy.tick():
        screen = pyboy.screen.image
        if screen.mode == 'RGBA':
            screen = screen.convert('RGB')

        frame = np.array(screen)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = model(frame, imgsz=320, conf=0.5, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                label = "Unknown"
                color = (255, 255, 255)

                if cls == 0:
                    label = "Player"
                    color = (255, 0, 0)
                elif cls == 1:
                    label = "NPC"
                    color = (0, 0, 255)
                elif cls == 2:
                    label = "Door"
                    color = (0, 255, 255)
                elif cls == 3:
                    label = "Sign"
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                text = f"{label} {conf:.0%}"
                (w, h), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                cv2.rectangle(frame, (x1, y1 - 15), (x1 + w, y1), color, -1)
                cv2.putText(frame, text, (x1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        big_frame = cv2.resize(frame, (640, 576),
                               interpolation=cv2.INTER_NEAREST)
        cv2.imshow("YOLO Vision v1", big_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pyboy.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
