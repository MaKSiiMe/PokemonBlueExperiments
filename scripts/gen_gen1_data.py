#!/usr/bin/env python3
"""Génère `src/pokeblue/knowledge/gen1_data/tables.py` depuis les données de pret/pokered.

Tables produites (commit épinglé dans `pokeblue.knowledge.pokered.source`) :
  - TYPE_NAMES, SPECIAL_TYPES_START  constants/type_constants.asm
  - TYPE_EFFECTS                     data/types/type_matchups.asm (ordre du jeu conservé)
  - MOVES                            data/moves/moves.asm
  - SPECIES                          data/pokemon/base_stats/*.asm, data/pokemon/dex_order.asm
  - MAPS                             constants/map_constants.asm
  - ITEMS                            constants/item_constants.asm (objets, étages, CT/CS)
  - EVENTS                           constants/event_constants.asm (indices de wEventFlags)
  - CHARMAP                          constants/charmap.asm (caractère → tuile)
  - FADE_PALETTES                    home/fade.asm (palettes rBGP/rOBP0/rOBP1 des fondus)

Usage :
    python scripts/gen_gen1_data.py           # télécharge (cache .cache/pokered) puis écrit
    python scripts/gen_gen1_data.py --check   # échoue si le fichier versionné est périmé
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from _codegen import ROOT, base_parser, write_or_check

from pokeblue.knowledge.pokered.asm import AsmConstants, AsmError, logical_lines, macro_args
from pokeblue.knowledge.pokered.source import POKERED_COMMIT, fetch_pokered, load_constants

OUT = ROOT / "src" / "pokeblue" / "knowledge" / "gen1_data" / "tables.py"


def _db_rows(path: Path, consts: AsmConstants, start_label: str | None = None) -> list[list[int]]:
    """Arguments évalués de chaque ligne `db` (après `start_label` si donné)."""
    rows, started = [], start_label is None
    for line in logical_lines(path):
        if not started:
            started = line.rstrip(":") == start_label
            continue
        args = macro_args(line, "db")
        if args is not None:
            rows.append([consts.eval(a) for a in args])
    return rows


def type_tables(root: Path, consts: AsmConstants, enumerated: list[str]):
    names = {consts[n]: n for n in enumerated}
    effects = []
    for row in _db_rows(root / "data/types/type_matchups.asm", consts, "TypeEffects"):
        if row == [-1]:
            break
        atk, dfn, factor = row
        if atk not in names or dfn not in names:
            raise AsmError(f"type inconnu dans TypeEffects : {row}")
        effects.append((atk, dfn, factor))
    return names, consts["SPECIAL"], effects


def moves_table(root: Path, consts: AsmConstants) -> dict[int, tuple]:
    moves = {}
    for line in logical_lines(root / "data/moves/moves.asm"):
        args = macro_args(line, "move")
        if args is None:
            continue
        name, effect, power, mtype, accuracy, pp = args
        consts.require(effect)
        moves[consts[name]] = (name, effect, consts.eval(power), consts.eval(mtype),
                               consts.eval(accuracy), consts.eval(pp))
    expected = set(range(1, consts["NUM_ATTACKS"] + 1))
    if set(moves) != expected:
        raise AsmError(f"moves.asm : IDs {sorted(expected ^ set(moves))} incohérents")
    return moves


def species_table(root: Path, consts: AsmConstants, enumerated: list[str]) -> dict[int, tuple]:
    # PokedexOrder : une ligne par ID interne à partir de 1 (0 = MissingNo.)
    dex_by_internal = {
        internal: row[0]
        for internal, row in enumerate(_db_rows(root / "data/pokemon/dex_order.asm", consts), 1)
        if row[0]
    }
    name_by_internal = {consts[n]: n for n in enumerated}

    base_by_dex = {}
    for path in sorted((root / "data/pokemon/base_stats").glob("*.asm")):
        rows = []
        for line in logical_lines(path):
            args = macro_args(line, "db")
            if args is not None:
                rows.append(args)
        dex = consts.eval(rows[0][0])
        stats = [consts.eval(a) for a in rows[1]]
        types = tuple(consts.eval(a) for a in rows[2])
        catch_rate, base_exp = consts.eval(rows[3][0]), consts.eval(rows[4][0])
        start_moves = tuple(m for m in (consts.eval(a) for a in rows[5]) if m)
        growth = rows[6][0]
        consts.require(growth)
        base_by_dex[dex] = (stats, types, catch_rate, base_exp, start_moves, growth)

    species = {}
    for internal, dex in sorted(dex_by_internal.items()):
        stats, types, catch_rate, base_exp, start_moves, growth = base_by_dex[dex]
        species[internal] = (name_by_internal[internal], dex, *stats, types,
                             catch_rate, base_exp, start_moves, growth)
    dexes = sorted(s[1] for s in species.values())
    if dexes != list(range(1, consts["NUM_POKEMON"] + 1)):
        raise AsmError("dex_order.asm : la correspondance ID interne → Pokédex n'est pas bijective")
    return species


_CHARMAP = re.compile(r'^\s*charmap\s+"((?:[^"\\]|\\.)+)",\s*\$([0-9A-Fa-f]{2})\b')


def charmap_table(root: Path) -> dict[str, int]:
    """Caractère (ou balise de contrôle `<...>`) → numéro de tuile, dans l'ordre du fichier."""
    table: dict[str, int] = {}
    for line in (root / "constants/charmap.asm").read_text(encoding="utf-8").splitlines():
        if m := _CHARMAP.match(line):
            table[m.group(1)] = int(m.group(2), 16)
    if not table:
        raise AsmError("charmap.asm : aucune entrée")
    return table


_FADE_PAL = re.compile(r"^FadePal(\d)::\s*dc\s+(.*)$")


def fade_palettes(root: Path) -> list[tuple[int, int, int]]:
    """FadePal1..8 : (rBGP, rOBP0, rOBP1). Macro `dc` : 4 valeurs de 2 bits par octet."""
    palettes = {}
    for line in logical_lines(root / "home/fade.asm"):
        if m := _FADE_PAL.match(line):
            crumbs = [int(v) for v in m.group(2).split(",")]
            packed = [
                (a << 6) | (b << 4) | (c << 2) | d
                for a, b, c, d in zip(*[iter(crumbs)] * 4, strict=True)
            ]
            palettes[int(m.group(1))] = tuple(packed)
    if sorted(palettes) != list(range(1, 9)):
        raise AsmError("home/fade.asm : FadePal1..8 attendus")
    return [palettes[i] for i in range(1, 9)]


def _unique_values(consts: AsmConstants, names: list[str], label: str) -> dict[int, str]:
    table: dict[int, str] = {}
    for name in names:
        value = consts[name]
        if value in table:
            raise AsmError(f"{label} : {name} et {table[value]} partagent la valeur {value:#x}")
        table[value] = name
    return table


def render(root: Path) -> str:
    consts, enumerated = load_constants(root)
    type_names, special_start, effects = type_tables(
        root, consts, enumerated["constants/type_constants.asm"])
    moves = moves_table(root, consts)
    species = species_table(root, consts, enumerated["constants/pokemon_constants.asm"])
    maps = {consts[n]: (n, consts[f"{n}_WIDTH"], consts[f"{n}_HEIGHT"])
            for n in enumerated["constants/map_constants.asm"]}
    items = _unique_values(consts, enumerated["constants/item_constants.asm"], "items")
    events = _unique_values(consts, enumerated["constants/event_constants.asm"], "events")
    charmap = charmap_table(root)
    fades = fade_palettes(root)

    out = [
        '"""Données Gen 1 de Pokémon Bleu — FICHIER GÉNÉRÉ, NE PAS MODIFIER.',
        "",
        f"Généré par `scripts/gen_gen1_data.py` depuis pret/pokered @ {POKERED_COMMIT}.",
        "Les identifiants sont ceux du jeu (octets RAM) ; les noms sont les constantes pokered.",
        '"""',
        "",
        "from pokeblue.knowledge.gen1_data.models import MapInfo, Move, Species",
        "",
        "# fmt: off",
        f'SOURCE_COMMIT = "{POKERED_COMMIT}"',
        "",
        "# constants/type_constants.asm",
        "TYPE_NAMES: dict[int, str] = {",
        *(f'    0x{tid:02X}: "{name}",' for tid, name in sorted(type_names.items())),
        "}",
        "# Types >= SPECIAL utilisent la stat Spécial ; les autres, Attaque/Défense.",
        f"SPECIAL_TYPES_START = 0x{special_start:02X}",
        "",
        "# data/types/type_matchups.asm — (attaquant, défenseur, facteur × 10), dans l'ordre",
        "# du jeu (20 = ×2, 5 = ×0,5, 0 = aucun effet ; absent = ×1).",
        "TYPE_EFFECTS: tuple[tuple[int, int, int], ...] = (",
        *(f"    (0x{a:02X}, 0x{d:02X}, {f}),  # {type_names[a]} → {type_names[d]}"
          for a, d, f in effects),
        ")",
        "",
        "# data/moves/moves.asm — Move(nom, effet, puissance, type, précision %, PP)",
        "MOVES: dict[int, Move] = {",
        *(f'    0x{mid:02X}: Move("{n}", "{e}", {p}, 0x{t:02X}, {a}, {pp}),'
          for mid, (n, e, p, t, a, pp) in sorted(moves.items())),
        "}",
        "",
        "# data/pokemon/base_stats/*.asm, indexé par ID interne (data/pokemon/dex_order.asm)",
        "# Species(nom, dex, pv, atq, déf, vit, spé, types, capture, exp, attaques niv. 1, croissance)",
        "SPECIES: dict[int, Species] = {",
    ]
    for sid, (name, dex, hp, atk, dfn, spd, spc, types, catch, exp, start, growth) in sorted(
            species.items()):
        moves_src = "(" + ", ".join(f"0x{m:02X}" for m in start) + ("," if len(start) == 1 else "") + ")"
        out.append(
            f'    0x{sid:02X}: Species("{name}", {dex}, {hp}, {atk}, {dfn}, {spd}, {spc}, '
            f'(0x{types[0]:02X}, 0x{types[1]:02X}), {catch}, {exp}, {moves_src}, "{growth}"),'
        )
    out += [
        "}",
        "",
        "# constants/map_constants.asm — MapInfo(nom, largeur, hauteur) en blocs",
        "MAPS: dict[int, MapInfo] = {",
        *(f'    0x{mid:02X}: MapInfo("{n}", {w}, {h}),' for mid, (n, w, h) in sorted(maps.items())),
        "}",
        "",
        "# constants/item_constants.asm — objets, étages d'ascenseur, CS (HM_*) et CT (TM_*)",
        "ITEMS: dict[int, str] = {",
        *(f'    0x{iid:02X}: "{n}",' for iid, n in sorted(items.items())),
        "}",
        "",
        "# constants/event_constants.asm — indice du drapeau dans wEventFlags",
        "EVENTS: dict[int, str] = {",
        *(f'    0x{eid:03X}: "{n}",' for eid, n in sorted(events.items())),
        "}",
        "",
        "# constants/charmap.asm — caractère (ou balise <...>) → tuile, dans l'ordre du fichier",
        "CHARMAP: dict[str, int] = {",
        *(f"    {json.dumps(c, ensure_ascii=False)}: 0x{t:02X}," for c, t in charmap.items()),
        "}",
        "",
        "# home/fade.asm — FadePal1..8 : (rBGP, rOBP0, rOBP1), du noir (1) au blanc (8).",
        "# Palette stable d'une carte : LoadGBPal lit FadePal4 décalée de wMapPalOffset octets.",
        "FADE_PALETTES: tuple[tuple[int, int, int], ...] = (",
        *(f"    (0x{b:02X}, 0x{o0:02X}, 0x{o1:02X}),  # FadePal{i}" for i, (b, o0, o1) in enumerate(fades, 1)),
        ")",
        "# fmt: on",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    args = base_parser(__doc__.splitlines()[1], OUT).parse_args()
    root = args.pokered or fetch_pokered(args.cache_dir)
    return write_or_check(args.out, render(root), args.check)


if __name__ == "__main__":
    sys.exit(main())
