"""`pokeblue overlay` : écran du jeu + GameState + mode détecté, en direct.

Mode interactif : la fenêtre PyBoy (SDL2) reçoit le clavier, une seconde fenêtre
OpenCV (extra `tools`) affiche le panneau. Mode `--out` : rejoue une recette sans
fenêtre et enregistre l'image du panneau (utile pour la doc et le débogage).
"""

from __future__ import annotations

import argparse
import unicodedata
from collections import deque
from pathlib import Path

from PIL import Image, ImageDraw

from pokeblue.emulator import Emulator
from pokeblue.knowledge.gen1_data import ITEMS, MAPS, MOVES, SPECIES, TYPE_NAMES
from pokeblue.state import ram_symbols as sym
from pokeblue.state.game_state import BattleMon, GameState
from pokeblue.state.mode_detector import Mode, detect_mode
from pokeblue.state.screen import Screen

SCALE = 3
PANEL_W = 420
LINE_H = 13
FACINGS = {
    sym.SPRITE_FACING_DOWN: "bas", sym.SPRITE_FACING_UP: "haut",
    sym.SPRITE_FACING_LEFT: "gauche", sym.SPRITE_FACING_RIGHT: "droite",
}
MODE_COLORS = {
    Mode.OVERWORLD: "darkgreen", Mode.DIALOG: "navy", Mode.MENU: "purple",
    Mode.BATTLE_MENU: "darkred", Mode.BATTLE_MOVE_MENU: "darkred",
    Mode.BATTLE_ANIM: "orangered", Mode.TRANSITION: "gray",
}
STATUS_BITS = ((sym.PSN, "PSN"), (sym.BRN, "BRL"), (sym.FRZ, "GEL"), (sym.PAR, "PAR"))
# La police par défaut de PIL est ASCII : on translittère le texte du panneau.
_ASCII = str.maketrans({"┌": "+", "┐": "+", "└": "+", "┘": "+", "─": "-", "│": "|",
                        "▶": ">", "▷": ">", "▼": "v", "×": "x", "→": "->", "¥": "Y"})


def _ascii(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.translate(_ASCII))
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.encode("ascii", "replace").decode("ascii")


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rom", type=Path, default=Path("ROMs/PokemonBlue.gb"))
    p.add_argument("--state", type=Path, default=Path("states/00_pallet_town.state"))
    p.add_argument("--recipe", default="", help="recette d'inputs rejouée avant l'affichage")
    p.add_argument("--out", type=Path, default=None,
                   help="enregistre le panneau dans ce PNG sans ouvrir de fenêtre")
    p.add_argument("--speed", type=int, default=1, help="vitesse d'émulation (0 = max)")


def _species(sid: int) -> str:
    return SPECIES[sid].name if sid in SPECIES else f"#{sid:02x}"


def _status(status: int) -> str:
    if status & sym.SLP_MASK:
        return f"SOM{status & sym.SLP_MASK}"
    return next((name for bit, name in STATUS_BITS if status >> bit & 1), "")


def _battle_mon_lines(label: str, mon: BattleMon) -> list[str]:
    types = "/".join(dict.fromkeys(TYPE_NAMES.get(t, "?") for t in mon.types))
    moves = ", ".join(f"{MOVES[m].name}({pp})" for m, pp in zip(mon.moves, mon.pp, strict=True)
                      if m in MOVES)
    mods = " ".join(f"{v - sym.BASE_STAT_LEVEL:+d}" for v in mon.stat_mods)
    return [
        f"{label} {_species(mon.species)} N{mon.level} PV {mon.hp}/{mon.max_hp} "
        f"{types} {_status(mon.status)}",
        f"   {moves}",
        f"   paliers atq/déf/vit/spé/préc/esq : {mods}",
    ]


def state_lines(state: GameState) -> list[str]:
    map_name = MAPS[state.map_id].name if state.map_id in MAPS else f"{state.map_id:#04x}"
    lines = [
        f"carte {map_name} ({state.x},{state.y}) face {FACINGS.get(state.facing, state.facing)}",
        f"argent {state.money}  badges {state.n_badges}/8  drapeaux {state.n_flags}"
        f"  pokédex {state.pokedex_owned.bit_count()}/{state.pokedex_seen.bit_count()}",
        "équipe :",
    ]
    for mon in state.party:
        lines.append(f"  {_species(mon.species)} N{mon.level} PV {mon.hp}/{mon.max_hp} "
                     f"{_status(mon.status)}")
    if state.bag:
        bag = ", ".join(f"{ITEMS.get(i, hex(i))}×{q}" for i, q in state.bag)
        lines.append(f"sac : {bag}")
    if state.battle:
        b = state.battle
        kind = "sauvage" if b.is_wild else f"dresseur, {b.enemy_party_count} Pokémon"
        lines.append(f"combat : {kind} (wBattleType {b.battle_type})")
        lines += _battle_mon_lines("  moi ", b.player)
        lines += _battle_mon_lines("  ennemi", b.enemy)
    lines.append(f"rBGP {state.bgp:#04x} hWY {state.window_y} wJoyIgnore {state.joy_ignore:#04x}")
    return lines


def render_panel(state: GameState, mode: Mode, screen: Image.Image,
                 events: list[str] | None = None) -> Image.Image:
    """Image composée : écran agrandi à gauche, informations à droite."""
    big = screen.resize((screen.width * SCALE, screen.height * SCALE), Image.NEAREST)
    panel = Image.new("RGB", (big.width + PANEL_W, big.height), "white")
    panel.paste(big, (0, 0))
    draw = ImageDraw.Draw(panel)
    x, y = big.width + 8, 6
    draw.text((x, y), f"MODE : {mode.value}", fill=MODE_COLORS[mode])
    y += 2 * LINE_H
    for line in state_lines(state):
        draw.text((x, y), _ascii(line)[:64], fill="black")
        y += LINE_H
    text = [r for r in Screen(state.tilemap).text_rows(" ") if r.strip()]
    if text:
        y += LINE_H // 2
        draw.text((x, y), _ascii("texte à l'écran :"), fill="gray")
        y += LINE_H
        for row in text[:12]:
            draw.text((x, y), _ascii(row.rstrip()), fill="gray")
            y += LINE_H
    if events:
        y += LINE_H // 2
        draw.text((x, y), _ascii("derniers événements :"), fill="darkblue")
        y += LINE_H
        for event in events[-8:]:
            draw.text((x, y), _ascii(event)[:64], fill="darkblue")
            y += LINE_H
    return panel


def run(args: argparse.Namespace) -> int:
    if args.out:
        with Emulator(args.rom) as emu:
            emu.load_state(args.state)
            emu.run(args.recipe)
            screen = emu.screen_image()
            state = GameState.from_memory(emu.snapshot())
            render_panel(state, detect_mode(state), screen).save(args.out)
        print(f"panneau enregistré : {args.out}")
        return 0

    try:
        import cv2
        import numpy as np
    except ImportError:
        print("OpenCV manquant : pip install -e '.[tools]'")
        return 1

    emu = Emulator(args.rom, window="SDL2", speed=args.speed)
    emu.load_state(args.state)
    emu.run(args.recipe)
    events: deque[str] = deque(maxlen=8)
    prev = GameState.from_memory(emu.snapshot())
    frame = 0
    try:
        while emu.pyboy.tick(1, True):
            frame += 1
            if frame % 4:
                continue
            state = GameState.from_memory(emu.snapshot())
            mode = detect_mode(state)
            events.extend(f"f{frame} {line}" for line in state.diff(prev).describe())
            prev = state
            panel = render_panel(state, mode, emu.pyboy.screen.image.convert("RGB"), list(events))
            cv2.imshow("pokeblue overlay", cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        emu.close()
        cv2.destroyAllWindows()
    return 0
