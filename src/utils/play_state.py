from pyboy import PyBoy
import os
import time

# ================= CONFIGURATION =================
# Mets ici le nom exact de ton fichier .state
STATE_TO_LOAD = "50_endgame.state" 
ROM_PATH = "PokemonBlue.gb"
# =================================================

def main():
    if not os.path.exists(ROM_PATH):
        print(f"❌ Erreur : ROM introuvable ({ROM_PATH})")
        return

    # 1. On lance l'émulateur
    print("🎮 Lancement de l'émulateur...")
    pyboy = PyBoy(ROM_PATH, window="SDL2", sound=True)
    pyboy.set_emulation_speed(1)

    # 2. PHASE DE CHAUFFE (CRUCIAL !)
    # On laisse l'émulateur faire 20 "ticks" (images) pour s'initialiser.
    # Si on charge le state tout de suite, il risque d'être écrasé par le boot du jeu.
    print("⏳ Initialisation du moteur (Warm-up)...")
    for _ in range(20):
        pyboy.tick()

    # 3. Maintenant on charge la sauvegarde
    if STATE_TO_LOAD and os.path.exists(STATE_TO_LOAD):
        print(f"📂 Injection de la sauvegarde : {STATE_TO_LOAD}")
        with open(STATE_TO_LOAD, "rb") as f:
            pyboy.load_state(f)
        print("✅ Sauvegarde chargée ! À toi de jouer.")
    else:
        print(f"⚠️ Fichier state '{STATE_TO_LOAD}' introuvable ou non défini.")

    # 4. Boucle de jeu
    while pyboy.tick():
        pass

    pyboy.stop()

if __name__ == "__main__":
    main()