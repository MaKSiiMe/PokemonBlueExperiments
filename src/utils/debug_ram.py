from pyboy import PyBoy
import time

# ⚠️ Vérifie que le chemin est bon par rapport à où tu lances le script
ROM_PATH = "PokemonBlue.gb" 

def main():
    # On lance en mode graphique pour que tu puisses bouger le perso avec le clavier
    pyboy = PyBoy(ROM_PATH, window="SDL2")
    pyboy.set_emulation_speed(1) # Vitesse normale
    
    print("🕵️  Lancement du scanner RAM...")
    print("👉 Déplace le joueur avec les flèches pour voir les valeurs changer !")

    # --- ADRESSES MÉMOIRE (POKEMON RED/BLUE US) ---
    # Ces adresses sont spécifiques à la version US.
    ADDR_X_POS = 0xD362
    ADDR_Y_POS = 0xD361
    ADDR_MAP_N = 0xD35E
    
    with pyboy:
        while pyboy.tick():
            # Lecture de la RAM
            x_pos = pyboy.memory[ADDR_X_POS]
            y_pos = pyboy.memory[ADDR_Y_POS]
            map_id = pyboy.memory[ADDR_MAP_N]
            
            # Affichage formaté (reviens à la ligne proprement)
            # \r permet d'écraser la ligne précédente pour faire un affichage dynamique
            print(f"📍 Position: ({x_pos}, {y_pos}) | 🗺️  Map ID: {map_id}   ", end="\r")
            
            # On ralentit un tout petit peu l'affichage console pour pas spammer
            # (Mais le jeu continue de tourner)
            pass

if __name__ == "__main__":
    main()
