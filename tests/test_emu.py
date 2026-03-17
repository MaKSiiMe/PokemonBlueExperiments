from pyboy import PyBoy

ROM_PATH = "ROMs/PokemonBlue.gb"


def main():
    print("🎮 Démarrage de PyBoy...")
    pyboy = PyBoy(ROM_PATH, window="SDL2", sound=False)

    print("✅ Émulateur chargé !")
    print("⚡ Vitesse simulée : Normale.")

    for i in range(500):
        pyboy.tick()
        if i % 100 == 0:
            print(f"Frame {i}/500")

    pyboy.stop()
    print("🛑 Test terminé avec succès.")


if __name__ == "__main__":
    main()
