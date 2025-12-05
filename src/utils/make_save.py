from pyboy import PyBoy


def main():
    print("🎮 Lance le jeu en mode manuel...")
    print("👉 Passe l'intro, choisis ton nom, "
          "et attends d'être dans la chambre.")
    print("💾 Quand tu es prêt, appuie sur 'S' pour sauvegarder.")
    print("❌ Ensuite, ferme la fenêtre.")

    pyboy = PyBoy("PokemonBlue.gb", window="SDL2")

    with pyboy:
        while pyboy.tick():
            pass

    print("Sauvegarde de l'état dans 'init.state'...")
    with open("init.state", "wb") as f:
        pyboy.save_state(f)
    print("✅ C'est bon !")


if __name__ == "__main__":
    main()
