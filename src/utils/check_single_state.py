import sys
import os
from pyboy import PyBoy

# Ce script est appelé par le nettoyeur.
# Il prend un nom de fichier en argument.
if len(sys.argv) < 2:
    sys.exit(1)

state_file = sys.argv[1]
rom_path = "PokemonBlue.gb"

try:
    # Mode sans échec : Pas de fenêtre, pas de son
    pyboy = PyBoy(rom_path, window="null", sound=False)
    pyboy.set_emulation_speed(0) # Vitesse max pour tester vite

    with open(state_file, "rb") as f:
        pyboy.load_state(f)

    # Le Test de Survie : On essaie de faire tourner 100 frames
    for _ in range(100):
        pyboy.tick()
    
    pyboy.stop()
    sys.exit(0) # Succès : On est vivant

except Exception as e:
    sys.exit(1) # Échec : On a planté
