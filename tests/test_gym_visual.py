import cv2
import numpy as np
import time
from src.emulator.pokemon_env import PokemonBlueEnv

MODEL_PATH = "models/yolo11_pokemon/weights/best.pt"
ROM_PATH = "PokemonBlue.gb"
INIT_STATE = "states/06_pallet_town.state"

def main():
    print("📺 Démarrage du Debug Visuel de l'Environnement...")
    
    env = PokemonBlueEnv(ROM_PATH, MODEL_PATH, init_state=INIT_STATE, headless=False, speed=1)
    obs, info = env.reset()

    print("✅ Environnement chargé. Appuie sur 'q' pour quitter.")

    try:
        while True:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            screen = env.pyboy.screen.image
            if screen.mode == 'RGBA': screen = screen.convert('RGB')
            frame = np.array(screen)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if obs[0] > 0:
                px, py = int(obs[0] * 160), int(obs[1] * 144)
                cv2.circle(frame, (px, py), 5, (255, 0, 0), -1)
                cv2.putText(frame, "Player", (px+5, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            if obs[2] > 0:
                dx, dy = int(obs[2] * 160), int(obs[3] * 144)
                cv2.circle(frame, (dx, dy), 5, (0, 255, 255), -1)
                cv2.line(frame, (px, py), (dx, dy), (0, 255, 255), 1)
                cv2.putText(frame, "Target", (dx+5, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            if obs[4] > 0:
                sx, sy = int(obs[4] * 160), int(obs[5] * 144)
                cv2.circle(frame, (sx, sy), 5, (255, 0, 255), -1)

            big_frame = cv2.resize(frame, (640, 576), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Ce que l'IA voit (Gym Obs)", big_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
