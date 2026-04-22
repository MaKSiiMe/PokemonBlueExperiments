"""
create_curriculum_states.py — Création interactive des savestates du curriculum.

Lance le jeu avec fenêtre SDL2 et te laisse jouer normalement.
Appuie sur Ctrl+C quand tu es à la position voulue : l'état est sauvegardé.

Usage :
    python utils/create_curriculum_states.py --name route1
    python utils/create_curriculum_states.py --name viridian --base states/01_route1.state
    python utils/create_curriculum_states.py --name pewter   --base states/02_viridian.state

États cibles recommandés :
    01_route1.state   — Entrée de Route 1 (juste après Bourg Palette)
    02_viridian.state — Entrée de Bourg des Eaux (premier Centre Pokémon)
    03_pewter.state   — Entrée d'Argenta (devant l'arène de Brock)

Notes :
    - La fenêtre SDL2 est nécessaire (ne fonctionne pas en headless WSL2 sans X11)
    - Sur WSL2 : lance depuis Windows Terminal avec VcXsrv ou WSLg activé
    - Vitesse réelle (1×) pour jouer confortablement
"""

import argparse
import os
import signal
import sys

ROM_PATH    = 'ROMs/PokemonBlue.gb'
STATES_DIR  = 'states'


def parse_args():
    p = argparse.ArgumentParser(description='Crée un savestate à une position donnée')
    p.add_argument('--name', required=True,
                   help="Nom du savestate (ex: route1 → states/01_route1.state)")
    p.add_argument('--base', default=None,
                   help='Savestate de départ (optionnel, sinon repart de Bourg Palette)')
    p.add_argument('--speed', type=int, default=1,
                   help='Vitesse de l\'émulateur (1=réel, 2=2×, etc.)')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(ROM_PATH):
        print(f"[Erreur] ROM introuvable : {ROM_PATH}")
        sys.exit(1)

    os.makedirs(STATES_DIR, exist_ok=True)

    # Déterminer le chemin de sortie
    # Si le nom contient déjà un chiffre préfixe (ex: "01_route1"), on garde tel quel
    if args.name[0].isdigit():
        out_path = os.path.join(STATES_DIR, f"{args.name}.state")
    else:
        out_path = os.path.join(STATES_DIR, f"{args.name}.state")

    from pyboy import PyBoy

    print(f"\n{'='*60}")
    print(f"  Création de : {out_path}")
    print(f"  Base        : {args.base or 'Bourg Palette (state initial)'}")
    print(f"  Vitesse     : {args.speed}×")
    print(f"{'='*60}")
    print()
    print("  ► Joue jusqu'à la position souhaitée")
    print("  ► Appuie sur Ctrl+C pour sauvegarder et quitter")
    print()

    pyboy = PyBoy(ROM_PATH, window='SDL2', sound=False)
    pyboy.set_emulation_speed(args.speed)

    # Charger l'état de base si fourni
    if args.base:
        if not os.path.exists(args.base):
            print(f"[Erreur] État de base introuvable : {args.base}")
            pyboy.stop()
            sys.exit(1)
        with open(args.base, 'rb') as f:
            pyboy.load_state(f)
        print(f"[OK] État de base chargé : {args.base}")
    else:
        # Chercher le state initial
        default = os.path.join(STATES_DIR, '00_pallet_town.state')
        if os.path.exists(default):
            with open(default, 'rb') as f:
                pyboy.load_state(f)
            print(f"[OK] Départ depuis : {default}")

    def save_and_exit(sig, frame):
        with open(out_path, 'wb') as f:
            pyboy.save_state(f)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"\n[OK] État sauvegardé : {out_path}  ({size_kb:.0f} KB)")
        print()
        print("  Pour utiliser le curriculum, lance :")
        print(f"  python run_agent.py --train --curriculum")
        pyboy.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, save_and_exit)

    print("[Jeu] Démarre... (Ctrl+C pour sauvegarder)\n")
    try:
        while True:
            pyboy.tick(1, render=True)
    except Exception:
        pass

    pyboy.stop()


if __name__ == '__main__':
    main()
