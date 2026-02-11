import sys
import os
import time
from src.emulator.pokemon_env import PokemonBlueEnv

MODEL_PATH = "models/yolo11_pokemon/weights/best.pt"
ROM_PATH = "PokemonBlue.gb"

def main():
    print(f"🧪 Démarrage du Test Gym avec le modèle : {MODEL_PATH}")

    try:
        env = PokemonBlueEnv(ROM_PATH, MODEL_PATH, headless=False, speed=1)
    except Exception as e:
        print(f"❌ Erreur au lancement : {e}")
        return

    print("🔄 Reset de l'environnement (Attente de l'intro)...")
    obs, info = env.reset()
    print(f"✅ Observation Initiale reçue : {obs}")
    
    print("\n🎮 Début de la marche aléatoire (100 pas)...")
    print("Légende : P=Player, D=Door, S=Sign\n")

    for step in range(100):
        action_idx = env.action_space.sample()
        action_name = env.valid_actions[action_idx]

        obs, reward, done, truncated, info = env.step(action_idx)

        p_x, p_y = obs[0], obs[1]
        d_x, d_y = obs[2], obs[3]
        s_x, s_y = obs[4], obs[5]

        log = f"Step {step:03d} | Action: {action_name.upper():<6} | P:({p_x:.2f}, {p_y:.2f})"
        
        if d_x > 0 or d_y > 0:
            log += f" | 🚪 DOOR ({d_x:.2f}, {d_y:.2f})"
        
        if s_x > 0 or s_y > 0:
            log += f" | 📜 SIGN ({s_x:.2f}, {s_y:.2f})"

        print(log)

    print("\n🛑 Fin du test.")
    env.close()

if __name__ == "__main__":
    main()
