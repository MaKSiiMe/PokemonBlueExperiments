"""Point d'entrée unique : `pokeblue <commande>`.

    pokeblue build-knowledge  régénère les données issues de pokered (--check : vérifie)
    pokeblue make-states   régénère les savestates d'un manifeste (configs/states/*.yaml)
    pokeblue overlay       fenêtre de debug : écran + GameState + mode détecté
    pokeblue run           une partie jouée par l'orchestrateur (baseline scriptée)
    pokeblue eval          N parties, rapport (jalons, actions par badge, échecs par skill)
"""

from __future__ import annotations

import argparse
import sys

from pokeblue.tools import build_knowledge, evaluate, make_states, overlay, play

COMMANDS = {
    "build-knowledge": (build_knowledge, "régénère ou vérifie les données issues de pokered"),
    "make-states": (make_states, "régénère des savestates à partir d'un manifeste de recettes"),
    "overlay": (overlay, "affiche l'écran, le GameState et le mode détecté"),
    "run": (play, "joue une partie avec l'orchestrateur"),
    "eval": (evaluate, "évalue l'orchestrateur sur N parties"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokeblue", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (module, help_text) in COMMANDS.items():
        module.add_arguments(sub.add_parser(name, help=help_text))
    args = parser.parse_args(argv)
    return COMMANDS[args.command][0].run(args)


if __name__ == "__main__":
    sys.exit(main())
