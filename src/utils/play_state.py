from pyboy import PyBoy
import os

STATE_TO_LOAD = "states/init.state"
ROM_PATH = "PokemonBlue.gb"


def main():
    if not os.path.exists(ROM_PATH):
        print(f"❌ Erreur : ROM introuvable ({ROM_PATH})")
        return

    print("🎮 Lancement de l'émulateur...")
    pyboy = PyBoy(ROM_PATH, window="SDL2", sound=True)
    pyboy.set_emulation_speed(1)

    print("⏳ Initialisation du moteur (Warm-up)...")
    for _ in range(20):
        pyboy.tick()

    if STATE_TO_LOAD and os.path.exists(STATE_TO_LOAD):
        print(f"📂 Injection de la sauvegarde : {STATE_TO_LOAD}")
        with open(STATE_TO_LOAD, "rb") as f:
            pyboy.load_state(f)
        print("✅ Sauvegarde chargée ! À toi de jouer.")
    else:
        print(f"⚠️ Fichier state '{STATE_TO_LOAD}' introuvable.")

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()
