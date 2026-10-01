"""Point d'entrée unique : `pokeblue <commande>`.

    pokeblue build-knowledge  régénère les données issues de pokered (--check : vérifie)
    pokeblue make-states   régénère les savestates d'un manifeste (configs/states/*.yaml)
    pokeblue overlay       fenêtre de debug : écran + GameState + mode détecté
"""

from __future__ import annotations

import argparse
import sys

from pokeblue.tools import build_knowledge, make_states, overlay

COMMANDS = {
    "build-knowledge": (build_knowledge, "régénère ou vérifie les données issues de pokered"),
    "make-states": (make_states, "régénère des savestates à partir d'un manifeste de recettes"),
    "overlay": (overlay, "affiche l'écran, le GameState et le mode détecté"),
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
