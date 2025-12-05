from pyboy import PyBoy
import os
import threading

ROM_PATH = "PokemonBlue.gb"
STATES_DIR = "states"

os.makedirs(STATES_DIR, exist_ok=True)

INIT_STATE = os.path.join(os.path.dirname(__file__), "..", "..", "init.state")


def input_thread(pyboy):
    print("\n--- INSTRUCTIONS ---")
    print("1. Joue normalement dans la fenêtre.")
    print("2. Quand tu es à un endroit cool, "
          "tape 's' + ENTRÉE dans ce terminal.")
    print("3. Tape 'exit' pour finir.")
    print("--------------------\n")

    save_dir = STATES_DIR

    while True:
        cmd = input("Commande (s=save, exit=quitter) : ").strip().lower()

        if cmd == "exit":
            pyboy.stop()
            break

        elif cmd == "s":
            name = input("Nom du checkpoint (ex: route1) : ").strip()
            if name:
                filename = os.path.join(save_dir, f"{name}.state")
                with open(filename, "wb") as f:
                    pyboy.save_state(f)
                print(f"✅ Sauvegardé : {filename}\n")


def main():
    print("🎮 Lancement du jeu - Crée des sauvegardes avec F1-F9")
    print(f"📁 Les states seront sauvegardées dans: {STATES_DIR}/")

    pyboy = PyBoy(ROM_PATH, window="SDL2", sound=False)
    pyboy.set_emulation_speed(1)

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    if not os.path.exists(ROM_PATH):
        print(f"❌ ROM introuvable : {ROM_PATH}")
        print("Assure-toi que PokemonBlue.gb est à la racine du projet.")
        exit(1)

    print("🎮 MODE CRÉATION DE CHECKPOINTS")

    pyboy = PyBoy(ROM_PATH, window="SDL2", sound=False)
    pyboy.set_emulation_speed(1)

    if os.path.exists(INIT_STATE):
        with open(INIT_STATE, "rb") as f:
            pyboy.load_state(f)
            print("✅ init.state chargée!")

    thread = threading.Thread(target=input_thread, args=(pyboy,), daemon=True)
    thread.start()

    while pyboy.tick():
        pass

    print("👋 Fin du script.")
