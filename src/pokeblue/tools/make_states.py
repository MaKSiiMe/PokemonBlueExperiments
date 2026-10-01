"""`pokeblue make-states` : régénère les savestates d'un manifeste, avec captures.

Pour chaque entrée : `<out>/<nom>.state`, `<out>/<nom>.png`, puis une planche
`<out>/sheet.png` (nom, mode étiqueté, mode détecté) qui sert à vérifier les
étiquettes à l'œil, et un rapport texte des écarts éventuels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from pokeblue.emulator import Emulator
from pokeblue.emulator.recipes import load_manifest, play
from pokeblue.state.game_state import GameState
from pokeblue.state.mode_detector import detect_mode

_TILE_W, _TILE_H, _CAPTION = 160, 144, 24
_COLUMNS = 4


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", type=Path, default=Path("configs/states/modes.yaml"))
    p.add_argument("--rom", type=Path, default=Path("ROMs/PokemonBlue.gb"))
    p.add_argument("--states", type=Path, default=Path("states"),
                   help="dossier des savestates de base")
    p.add_argument("--out", type=Path, default=Path("states/modes"))
    p.add_argument("--only", nargs="*", default=None, help="noms d'entrées à générer")


def _caption(image: Image.Image, lines: list[tuple[str, str]]) -> Image.Image:
    tile = Image.new("RGB", (_TILE_W, _TILE_H + _CAPTION), "white")
    tile.paste(image, (0, _CAPTION))
    draw = ImageDraw.Draw(tile)
    for i, (text, color) in enumerate(lines):
        draw.text((2, 1 + 11 * i), text[:30], fill=color)
    return tile


def run(args: argparse.Namespace) -> int:
    entries = load_manifest(args.manifest)
    if args.only:
        entries = [e for e in entries if e.name in set(args.only)]
    args.out.mkdir(parents=True, exist_ok=True)

    tiles, mismatches = [], []
    with Emulator(args.rom) as emu:
        for entry in entries:
            play(emu, entry, args.states)
            emu.save_state_to(args.out / f"{entry.name}.state")
            image = emu.screen_image()   # avance d'une frame rendue
            image.resize((_TILE_W * 2, _TILE_H * 2), Image.NEAREST).save(
                args.out / f"{entry.name}.png")
            detected = detect_mode(GameState.from_memory(emu.snapshot())).value
            ok = entry.mode is None or detected == entry.mode
            if not ok:
                mismatches.append(f"{entry.name}: étiqueté {entry.mode}, détecté {detected}")
            print(f"{'ok ' if ok else 'ÉCART'} {entry.name:28s} {entry.mode or '-':18s} {detected}")
            tiles.append(_caption(image, [
                (entry.name, "black"),
                (f"{entry.mode} / {detected}", "darkgreen" if ok else "red"),
            ]))

    rows = (len(tiles) + _COLUMNS - 1) // _COLUMNS
    sheet = Image.new("RGB", (_COLUMNS * (_TILE_W + 4), rows * (_TILE_H + _CAPTION + 4)), "gray")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % _COLUMNS) * (_TILE_W + 4), (i // _COLUMNS) * (_TILE_H + _CAPTION + 4)))
    sheet.save(args.out / "sheet.png")
    print(f"{len(tiles)} savestates → {args.out}/ ; planche : {args.out / 'sheet.png'}")
    for line in mismatches:
        print("  " + line)
    return 1 if mismatches else 0
