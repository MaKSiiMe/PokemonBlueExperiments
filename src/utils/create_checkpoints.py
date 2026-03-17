"""
create_checkpoints.py — Génère des save states programmatiquement.

Deux modes :
  --auto   : exécute toutes les recettes définies dans RECIPES (headless)
  --manual : lance le jeu depuis un state existant avec fenêtre SDL2,
             tape 's' pour sauvegarder, 'exit' pour quitter.

Exemples :
  python src/utils/create_checkpoints.py --auto
  python src/utils/create_checkpoints.py --manual --from states/01_chambre.state
"""

import argparse
import os
import threading

from pyboy import PyBoy

ROM_PATH   = 'ROMs/PokemonBlue.gb'
STATES_DIR = 'states'
os.makedirs(STATES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# RECETTES : chaque entrée crée un state automatiquement depuis un state parent.
# 'moves' = liste de (direction, nb_répétitions)
# Le state est sauvegardé après les moves + 60 ticks de stabilisation.
# ---------------------------------------------------------------------------
RECIPES = [
    {
        'name':  '02_maison_1f',
        'from':  'states/01_chambre.state',
        'moves': [
            ('up',    1),
            ('right', 1),
            ('up',    1),
            ('right', 1),
            ('up',    1),
            ('right', 1),   # franchit l'escalier → map 0x25
        ],
        'settle_ticks': 120,   # laisser la transition se terminer
        'desc': 'Pied des escaliers, maison 1F (map 0x25)',
    },
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

TICKS_HOLD    = 20   # ticks bouton maintenu
TICKS_RELEASE = 10   # ticks après relâchement


def _move(pyboy, direction, n=1):
    for _ in range(n):
        pyboy.button_press(direction)
        for _ in range(TICKS_HOLD):
            pyboy.tick()
        pyboy.button_release(direction)
        for _ in range(TICKS_RELEASE):
            pyboy.tick()


def _state_path(name):
    if not name.endswith('.state'):
        name += '.state'
    return os.path.join(STATES_DIR, os.path.basename(name))


# ---------------------------------------------------------------------------
# MODE AUTO
# ---------------------------------------------------------------------------

def run_auto():
    pyboy = PyBoy(ROM_PATH, window='null', sound=False)
    pyboy.set_emulation_speed(0)

    for recipe in RECIPES:
        src = recipe['from']
        if not os.path.exists(src):
            print(f'[skip] {src} introuvable')
            continue

        with open(src, 'rb') as f:
            pyboy.load_state(f)
        for _ in range(60):
            pyboy.tick()

        for direction, n in recipe['moves']:
            _move(pyboy, direction, n)

        for _ in range(recipe.get('settle_ticks', 60)):
            pyboy.tick()

        out = _state_path(recipe['name'])
        with open(out, 'wb') as f:
            pyboy.save_state(f)

        mid = pyboy.memory[0xD35E]
        x   = pyboy.memory[0xD362]
        y   = pyboy.memory[0xD361]
        print(f"[OK] {out}  —  map=0x{mid:02X}  ({x},{y})  {recipe['desc']}")

    pyboy.stop()


# ---------------------------------------------------------------------------
# MODE MANUEL
# ---------------------------------------------------------------------------

def _input_thread(pyboy):
    print('\n--- INSTRUCTIONS ---')
    print("  s     → sauvegarder la position courante")
    print("  exit  → quitter")
    print('--------------------\n')
    while True:
        cmd = input('Commande : ').strip().lower()
        if cmd == 'exit':
            pyboy.stop()
            break
        elif cmd == 's':
            name = input('Nom du state (sans .state) : ').strip()
            if name:
                path = _state_path(name)
                with open(path, 'wb') as f:
                    pyboy.save_state(f)
                mid = pyboy.memory[0xD35E]
                x, y = pyboy.memory[0xD362], pyboy.memory[0xD361]
                print(f'Sauvegardé : {path}  (map=0x{mid:02X}  x={x}  y={y})\n')


def run_manual(from_state: str):
    pyboy = PyBoy(ROM_PATH, window='SDL2', sound=False)
    pyboy.set_emulation_speed(1)

    if from_state and os.path.exists(from_state):
        with open(from_state, 'rb') as f:
            pyboy.load_state(f)
        for _ in range(60):
            pyboy.tick()
        print(f'State chargé : {from_state}')
    else:
        print('Démarrage à froid (aucun state fourni)')

    thread = threading.Thread(target=_input_thread, args=(pyboy,), daemon=True)
    thread.start()

    while pyboy.tick():
        pass

    print('Fin.')


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--auto',   action='store_true', help='Générer tous les states via les recettes')
    group.add_argument('--manual', action='store_true', help='Lancer le jeu manuellement pour sauvegarder')
    parser.add_argument('--from', dest='from_state', default=None,
                        help='State de départ pour le mode --manual')
    args = parser.parse_args()

    if not os.path.exists(ROM_PATH):
        print(f'ROM introuvable : {ROM_PATH}')
        exit(1)

    if args.auto:
        run_auto()
    else:
        run_manual(args.from_state)
