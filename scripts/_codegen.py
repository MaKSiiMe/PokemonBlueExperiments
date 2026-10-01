"""Utilitaires communs aux générateurs `scripts/gen_*.py`."""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / ".cache" / "pokered"


def base_parser(description: str, default_out: Path) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                   help="cache des sources pokered téléchargées (défaut : .cache/pokered)")
    p.add_argument("--pokered", type=Path, default=None,
                   help="checkout local de pret/pokered au commit épinglé (évite le téléchargement)")
    p.add_argument("--out", type=Path, default=default_out, help="fichier Python généré")
    p.add_argument("--check", action="store_true",
                   help="ne rien écrire ; échoue si le fichier versionné diffère de la génération")
    return p


def write_or_check(path: Path, content: str, check: bool) -> int:
    """Écrit `content` dans `path`, ou en mode --check compare et retourne 1 si différent."""
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    if check:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current == content:
            print(f"{rel} : à jour")
            return 0
        diff = difflib.unified_diff(current.splitlines(), content.splitlines(),
                                    "versionné", "généré", lineterm="", n=1)
        sys.stdout.writelines(line + "\n" for line in list(diff)[:40])
        print(f"{rel} : PÉRIMÉ — relancer le générateur sans --check")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"{rel} : écrit ({content.count(chr(10))} lignes)")
    return 0
