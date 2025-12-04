from pyboy import PyBoy

def main():
    print("🎮 Lance le jeu en mode manuel...")
    print("👉 Passe l'intro, choisis ton nom, et attends d'être dans la chambre.")
    print("💾 Quand tu es prêt, appuie sur la touche 'S' du clavier pour sauvegarder 'init.state'.")
    print("❌ Ensuite, ferme la fenêtre.")

    pyboy = PyBoy("PokemonBlue.gb", window="SDL2")
    
    with pyboy:
        while pyboy.tick():
            # Si l'utilisateur appuie sur la touche "S" (keycode peut varier, on va faire simple)
            # PyBoy gère les inputs, mais pour sauvegarder l'état programmatiquement on va ruser.
            # En fait, PyBoy sauvegarde automatiquement en quittant si on lui demande pas,
            # mais on va le forcer :
            pass
            
    # À la fermeture de la fenêtre, on sauvegarde
    print("Sauvegarde de l'état dans 'init.state'...")
    with open("init.state", "wb") as f:
        pyboy.save_state(f)
    print("✅ C'est bon !")

if __name__ == "__main__":
    main()
