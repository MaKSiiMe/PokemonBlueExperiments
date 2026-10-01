"""Sources pret/pokered épinglées.

Les symboles RAM et les données de jeu viennent du **même** état du code :
la branche `symbols` de pokered est reconstruite à chaque commit de `master`,
et `SYMBOLS_COMMIT` est le build de `POKERED_COMMIT`.

Pour monter de version : choisir un commit de `symbols`, relever le commit de
`master` dont il est issu (même message de commit), mettre à jour les trois
constantes ci-dessous, puis relancer les générateurs de `scripts/`.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from pokeblue.knowledge.pokered.asm import AsmConstants

POKERED_REPO = "pret/pokered"

# Fichiers de constantes utiles, dans l'ordre de `includes.asm` (dépendances incluses).
CONSTANT_FILES = (
    "constants/ram_constants.asm",
    "constants/type_constants.asm",
    "constants/battle_constants.asm",
    "constants/move_constants.asm",
    "constants/move_effect_constants.asm",
    "constants/item_constants.asm",
    "constants/pokemon_constants.asm",
    "constants/pokedex_constants.asm",
    "constants/pokemon_data_constants.asm",
    "constants/trainer_constants.asm",
    "constants/sprite_constants.asm",
    "constants/sprite_data_constants.asm",
    "constants/map_constants.asm",
    "constants/map_data_constants.asm",
    "constants/toggle_constants.asm",
    "constants/tileset_constants.asm",
    "constants/event_constants.asm",
    "constants/text_constants.asm",
    "constants/menu_constants.asm",
)

# master — "Use constants for `wIsInBattle` values (#600)", 2026-08-27
POKERED_COMMIT = "a1a22aaf84d1675bcdbaeb194592379d586d838e"
# symbols — build de POKERED_COMMIT
SYMBOLS_COMMIT = "3f618d59edf43918f48f5e558c34e04cb2fc5619"
SYM_FILE = "pokeblue.sym"
SYM_SHA256 = "fe3a2394d98f5a9d34b1241a383eafe3e78adcc69933716a903d07af629be6ea"

_RAW_URL = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"
_TARBALL_URL = "https://codeload.github.com/{repo}/tar.gz/{ref}"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_sym(cache_dir: Path) -> Path:
    """Retourne le chemin de `pokeblue.sym` épinglé, téléchargé si absent du cache.

    Raises:
        ValueError: si le fichier ne correspond pas à `SYM_SHA256`.
    """
    path = cache_dir / f"symbols-{SYMBOLS_COMMIT}" / SYM_FILE
    if not path.exists():
        _download(_RAW_URL.format(repo=POKERED_REPO, ref=SYMBOLS_COMMIT, path=SYM_FILE), path)
    digest = sha256_of(path)
    if digest != SYM_SHA256:
        raise ValueError(f"{path}: sha256 {digest} != {SYM_SHA256} attendu")
    return path


def fetch_pokered(cache_dir: Path) -> Path:
    """Retourne la racine d'un checkout de pokered à `POKERED_COMMIT`, téléchargé si absent."""
    root = cache_dir / f"pokered-{POKERED_COMMIT}"
    if (root / "constants").is_dir():
        return root
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
        archive = Path(tmp) / "pokered.tar.gz"
        _download(_TARBALL_URL.format(repo=POKERED_REPO, ref=POKERED_COMMIT), archive)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        # L'archive GitHub contient un unique dossier racine `pokered-<sha>/`.
        (extracted,) = (p for p in Path(tmp).iterdir() if p.is_dir())
        extracted.replace(root)
    return root


def load_constants(root: Path) -> tuple[AsmConstants, dict[str, list[str]]]:
    """Évalue `CONSTANT_FILES` d'un checkout pokered.

    Returns:
        (constantes, noms énumérés par fichier — `const`/`map_const`… dans l'ordre)
    """
    consts = AsmConstants()
    enumerated = {rel: consts.load(root / rel) for rel in CONSTANT_FILES}
    return consts, enumerated
