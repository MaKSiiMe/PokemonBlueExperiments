from pyboy import PyBoy

# Assure-toi d'avoir ta ROM renommée "PokemonBlue.gb" dans le dossier racine
ROM_PATH = "PokemonBlue.gb"

def main():
    print("🎮 Démarrage de PyBoy...")
    # window="SDL2" permet de voir l'écran. Mets "headless" pour cacher la fenêtre.
    pyboy = PyBoy(ROM_PATH, window="SDL2", sound=False)

    print("✅ Émulateur chargé !")
    print("⚡ Vitesse simulée : Normale. Appuyez sur la fenêtre pour focus.")

    # Boucle simple pour faire tourner le jeu 500 frames
    for i in range(500):
        pyboy.tick()
        if i % 100 == 0:
            print(f"Frame {i}/500")

    pyboy.stop()
    print("🛑 Test terminé avec succès.")

if __name__ == "__main__":
    main()
