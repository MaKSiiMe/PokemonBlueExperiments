"""`pokeblue build-knowledge` : régénère (ou vérifie) les données issues de pokered.

Lance, depuis la racine du dépôt, les trois générateurs de `scripts/` :
symboles RAM, données Gen 1 et cartes. Les sources pokered épinglées sont
téléchargées une fois dans `.cache/pokered`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

GENERATORS = ("gen_ram_symbols.py", "gen_gen1_data.py", "gen_maps.py")


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--check", action="store_true",
                   help="ne rien écrire ; échoue si un fichier généré est périmé")
    p.add_argument("--root", type=Path, default=Path.cwd(), help="racine du dépôt")


def run(args: argparse.Namespace) -> int:
    scripts = args.root / "scripts"
    if not (scripts / GENERATORS[0]).exists():
        print(f"générateurs introuvables dans {scripts} (lancer depuis la racine du dépôt)")
        return 1
    env = {**os.environ, "PYTHONPATH": str(args.root / "src")}
    status = 0
    for script in GENERATORS:
        cmd = [sys.executable, str(scripts / script)] + (["--check"] if args.check else [])
        status |= subprocess.run(cmd, cwd=args.root, env=env).returncode
    return status
