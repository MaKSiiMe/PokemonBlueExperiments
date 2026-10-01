"""Règles de projet vérifiées automatiquement.

1. Aucune adresse RAM (WRAM, registres IO, HRAM) n'est écrite en dur hors de
   `src/pokeblue/state/ram_symbols.py` : tout passe par les symboles générés.
   Seules exceptions : les lignes marquées `# allow-ram-literal` (bornes de régions).
2. Les fichiers générés correspondent à leurs générateurs (sauté si les sources
   pokered ne sont pas dans le cache local `.cache/pokered`).
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from pokeblue.state import ram_symbols as sym

ROOT = Path(__file__).resolve().parents[1]
GENERATED_RAM = ROOT / "src" / "pokeblue" / "state" / "ram_symbols.py"
SCANNED_DIRS = ("src", "tests", "scripts", "utils")
EXCLUDED_DIRS = {".venv", ".cache", "__pycache__", "archive"}
ALLOW_MARKER = "# allow-ram-literal"

_HEX = re.compile(r"0[xX]([0-9A-Fa-f]{4})(?![0-9A-Fa-f])")


def _is_ram_address(value: int) -> bool:
    """WRAM, registres IO et HRAM (la VRAM n'est pas surveillée : 0x8000 est un masque courant)."""
    return sym.WRAM_START <= value <= sym.WRAM_END or sym.IO_START <= value <= sym.HIGH_END


def _python_files():
    yield from ROOT.glob("*.py")
    for top in SCANNED_DIRS:
        for path in (ROOT / top).rglob("*.py"):
            if not EXCLUDED_DIRS & set(path.relative_to(ROOT).parts):
                yield path


def test_no_hardcoded_ram_addresses():
    offenders = []
    for path in _python_files():
        if path == GENERATED_RAM:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if ALLOW_MARKER in line:
                continue
            for m in _HEX.finditer(line):
                if _is_ram_address(int(m.group(1), 16)):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "Adresses RAM en dur — utiliser pokeblue.state.ram_symbols :\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize("script", ["gen_ram_symbols.py", "gen_gen1_data.py", "gen_maps.py"])
def test_generated_files_are_up_to_date(script):
    if not any((ROOT / ".cache" / "pokered").glob("pokered-*")):
        pytest.skip("sources pokered absentes du cache (lancer le générateur une fois)")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--check"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
