#!/usr/bin/env python3
"""Génère les données de cartes de `src/pokeblue/knowledge/data/` depuis pret/pokered.

Produit :
  - `maps/<CARTE>.json` (une par carte) : dimensions, tileset, connexions, warps,
    panneaux, objets (dresseurs, objets, état initial des objets activables),
    rencontres sauvages, numéro d'arène, et la tuile représentative de chaque case ;
  - `tilesets.json` : tuiles praticables, herbe, comptoirs, et règles de déplacement
    (corniches, eau, arbres à couper, paires de tuiles interdites).

Une case de déplacement fait 2×2 tuiles ; le moteur teste la tuile en bas à gauche
(GetTileAndCoordsInFrontOfPlayer), d'où la tuile « représentative » stockée par case.

Usage :
    python scripts/gen_maps.py           # télécharge (cache .cache/pokered) puis écrit
    python scripts/gen_maps.py --check   # échoue si les fichiers versionnés sont périmés
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from _codegen import ROOT, base_parser

from pokeblue.knowledge.pokered.asm import AsmConstants, AsmError, logical_lines, macro_args
from pokeblue.knowledge.pokered.source import POKERED_COMMIT, fetch_pokered, load_constants

OUT = ROOT / "src" / "pokeblue" / "knowledge" / "data"
BLOCK_TILES = 4          # un bloc = 4×4 tuiles = 2×2 cases
SQUARE_TILES = 2

# Règles codées dans le moteur (pas de table de données), avec leur source.
ENGINE_RULES = {
    "water_tiles": {
        "tiles": [0x14, 0x32, 0x48],
        "not_in_tilesets": {"0x32": ["SHIP_PORT"], "0x48": ["SHIP_PORT"]},
        "source": "engine/items/item_effects.asm IsNextTileShoreOrWater ; "
                  "home/overworld.asm CollisionCheckOnWater",
    },
    "cut_tree_tiles": {
        "OVERWORLD": [0x3D], "GYM": [0x50],
        "source": "engine/overworld/cut.asm UsedCut",
    },
    "ledges_only_in": ["OVERWORLD"],
    "ledges_source": "engine/overworld/ledges.asm HandleLedges",
}


def _labels_and_lines(path: Path) -> list[str]:
    return list(logical_lines(path))


def _blocks_paths(root: Path) -> dict[str, str]:
    """Étiquette `X_Blocks` → fichier .blk (étiquettes empilées comprises)."""
    paths, pending = {}, []
    for line in logical_lines(root / "maps.asm"):
        m = re.match(r"^(\w+_Blocks):\s*(.*)$", line)
        if m:
            pending.append(m.group(1))
            line = m.group(2)
        inc = re.match(r'^INCBIN\s+"([^"]+)"', line)
        if inc and pending:
            for label in pending:
                paths[label] = inc.group(1)
            pending = []
    return paths


def _tilesets(root: Path, consts: AsmConstants, tileset_names: list[str]) -> dict:
    """Tileset → blockset (.bst), tuiles praticables, herbe, comptoirs."""
    gfx = {}
    pending = []
    for line in logical_lines(root / "gfx/tilesets.asm"):
        m = re.match(r"^(\w+)_Block::\s*(.*)$", line)
        if m:
            pending.append(m.group(1))
            line = m.group(2)
        inc = re.match(r'^INCBIN\s+"([^"]+\.bst)"', line)
        if inc and pending:
            for name in pending:
                gfx[name] = inc.group(1)
            pending = []

    coll, pending = {}, []
    for line in logical_lines(root / "data/tilesets/collision_tile_ids.asm"):
        m = re.match(r"^(\w+)_Coll::$", line)
        if m:
            pending.append(m.group(1))
            continue
        args = macro_args(line, "coll_tiles")
        if args is not None and pending:
            for name in pending:
                coll[name] = [consts.eval(a) for a in args]
            pending = []

    headers = [args for line in logical_lines(root / "data/tilesets/tileset_headers.asm")
               if (args := macro_args(line, "tileset")) is not None]
    if len(headers) != len(tileset_names):
        raise AsmError("tileset_headers.asm : nombre de tilesets incohérent")
    tilesets = {}
    for const_name, (label, c1, c2, c3, grass, _anim) in zip(tileset_names, headers, strict=True):
        counters = [v for v in (consts.eval(c1), consts.eval(c2), consts.eval(c3)) if v >= 0]
        grass_tile = consts.eval(grass)
        tilesets[const_name] = {
            "id": consts[const_name],
            "blockset": gfx[label],
            "passable": coll[label],
            "grass_tile": grass_tile if grass_tile >= 0 else None,
            "counter_tiles": counters,
        }
    return tilesets


def _db_table(path: Path, consts: AsmConstants, label: str, columns: int | None = None) -> list[list[int]]:
    """Lignes `db` après `label` jusqu'à `db -1` ; seules les `columns` premières sont évaluées."""
    rows, started = [], False
    for line in logical_lines(path):
        if line.rstrip(":") == label:
            started = True
            continue
        if started:
            if re.fullmatch(r"\w+::?", line):
                break
            args = macro_args(line, "db")
            if args is not None:
                if args == ["-1"]:
                    break
                rows.append([consts.eval(a) for a in args[:columns]])
    return rows


def _movement_rules(root: Path, consts: AsmConstants, tileset_by_id: dict[int, str]) -> dict:
    facing = {consts[f"SPRITE_FACING_{d}"]: d.lower() for d in ("DOWN", "UP", "LEFT", "RIGHT")}
    ledges = [
        {"direction": facing[d], "from_tile": standing, "ledge_tile": ledge}
        for d, standing, ledge in _db_table(          # 4e colonne : touche (PAD_*), inutile ici
            root / "data/tilesets/ledge_tiles.asm", consts, "LedgeTiles", columns=3)
    ]
    pairs = {}
    for kind, label in (("land", "TilePairCollisionsLand"), ("water", "TilePairCollisionsWater")):
        pairs[kind] = [
            {"tileset": tileset_by_id[ts], "tiles": [a, b]}
            for ts, a, b in _db_table(root / "data/tilesets/pair_collision_tile_ids.asm", consts, label)
        ]
    water_tilesets = [tileset_by_id[row[0]] for row in _db_table(
        root / "data/tilesets/water_tilesets.asm", consts, "WaterTilesets")]
    return {
        "ledges": ledges,
        "tile_pair_collisions": pairs,
        "water_tilesets": water_tilesets,
        **ENGINE_RULES,
    }


def _object_file(root: Path, label: str) -> Path:
    path = root / "data/maps/objects" / f"{label}.asm"
    if not path.exists():
        raise AsmError(f"objets introuvables pour {label}")
    return path


def _parse_objects(root: Path, label: str, consts: AsmConstants) -> dict:
    data = {"border_block": None, "warps": [], "signs": [], "objects": [], "object_names": []}
    for line in logical_lines(_object_file(root, label)):
        word = line.split(None, 1)[0]
        args = macro_args(line, word) or []
        if word == "db" and data["border_block"] is None:
            data["border_block"] = consts.eval(args[0])
        elif word in ("const_export", "const"):
            data["object_names"].append(args[0])
        elif word == "warp_event":
            x, y, dest, warp = args
            data["warps"].append({
                "x": consts.eval(x), "y": consts.eval(y),
                "map": dest, "warp": consts.eval(warp),   # n° de warp d'arrivée, à partir de 1
            })
        elif word == "bg_event":
            data["signs"].append({"x": consts.eval(args[0]), "y": consts.eval(args[1]), "text": args[2]})
        elif word == "object_event":
            obj = {
                "x": consts.eval(args[0]), "y": consts.eval(args[1]),
                "sprite": args[2], "movement": args[3], "range_or_direction": args[4],
                "text": args[5], "trainer": None, "item": None,
            }
            if len(args) == 8:
                obj["trainer"] = {"class": args[6].removeprefix("OPP_"), "party": consts.eval(args[7])}
            elif len(args) == 7:
                obj["item"] = args[6]
            data["objects"].append(obj)
    names = data["object_names"]
    for index, obj in enumerate(data["objects"], 1):
        obj["index"] = index
        obj["name"] = names[index - 1] if index <= len(names) else None
    return data


def _toggles(root: Path, consts: AsmConstants, object_index: dict[tuple[str, str], int]) -> list:
    """ToggleableObjectStates, dans l'ordre des constantes TOGGLE_* (bit de wToggleableObjectFlags)."""
    entries, current_map = [], None
    for line in logical_lines(root / "data/maps/toggleable_objects.asm"):
        word = line.split(None, 1)[0]
        args = macro_args(line, word) or []
        if word == "toggleable_objects_for":
            current_map = args[0]
        elif word == "toggle_object_state":
            # Comme le jeu : la carte est celle de la section en cours (toggle_map_id),
            # l'objet est une constante d'objet de cette carte ou un numéro brut.
            obj, state = args
            index = object_index.get((current_map, obj))
            if index is None:
                index = consts.eval(obj)
            entries.append({
                "map": current_map, "object": index,
                "object_name": obj if (current_map, obj) in object_index else None,
                "initially_visible": consts.eval(state) == consts["ON"],
            })
    return entries


def _wild(root: Path, consts: AsmConstants, map_names: dict[int, str]) -> dict[str, dict]:
    pointers = [args[0] for line in logical_lines(root / "data/wild/grass_water.asm")
                if (args := macro_args(line, "dw")) is not None]
    tables: dict[str, dict] = {}
    for path in sorted((root / "data/wild/maps").glob("*.asm")):
        label, table, section = None, None, None
        for line in logical_lines(path):
            if re.fullmatch(r"\w+:", line):
                label = line[:-1]
                table = tables[label] = {"grass_rate": 0, "grass": [], "water_rate": 0, "water": []}
                continue
            word = line.split(None, 1)[0]
            args = macro_args(line, word) or []
            if word in ("def_grass_wildmons", "def_water_wildmons"):
                section = "grass" if "grass" in word else "water"
                table[f"{section}_rate"] = consts.eval(args[0])
            elif word == "db" and section:
                table[section].append([consts.eval(args[0]), args[1]])
            elif word.startswith("end_"):
                section = None
    wild = {}
    for map_id, label in enumerate(pointers):
        if label in tables and map_id in map_names and label != "NothingWildMons":
            wild[map_names[map_id]] = tables[label]
    return wild


def _gym_leader_numbers(root: Path) -> dict[str, int]:
    """scripts/<Carte>.asm : `ld a, $N` puis `ld [wGymLeaderNo], a`."""
    numbers = {}
    for path in (root / "scripts").glob("*.asm"):
        lines = list(logical_lines(path))
        for i, line in enumerate(lines):
            if line.replace(" ", "") == "ld[wGymLeaderNo],a" and i:
                m = re.fullmatch(r"ld a, \$([0-9A-Fa-f]+)", lines[i - 1])
                if m:
                    numbers[path.stem] = int(m.group(1), 16)
    return numbers


def render(root: Path) -> dict[str, str]:
    consts, enumerated = load_constants(root)
    tileset_names = enumerated["constants/tileset_constants.asm"]
    tileset_by_id = {consts[n]: n for n in tileset_names}
    tilesets = _tilesets(root, consts, tileset_names)
    blocks_paths = _blocks_paths(root)
    map_names = {consts[n]: n for n in enumerated["constants/map_constants.asm"]}
    gym_numbers = _gym_leader_numbers(root)

    maps, object_index = {}, {}
    for header in sorted((root / "data/maps/headers").glob("*.asm")):
        info, connections = None, []
        for line in logical_lines(header):
            word = line.split(None, 1)[0]
            args = macro_args(line, word) or []
            if word == "map_header":
                info = {"label": args[0], "name": args[1], "tileset": args[2]}
            elif word == "connection":
                connections.append({"direction": args[0], "map": args[2], "offset": consts.eval(args[3])})
        label, name = info["label"], info["name"]
        width, height = consts[f"{name}_WIDTH"], consts[f"{name}_HEIGHT"]
        objects = _parse_objects(root, label, consts)
        blk = (root / blocks_paths[f"{label}_Blocks"]).read_bytes()
        missing = width * height - len(blk)
        if not 0 <= missing <= width:
            raise AsmError(f"{label} : .blk de {len(blk)} octets pour {width}×{height} blocs")
        # UndergroundPathNorthSouth.blk ne fait que 4×23 blocs (voir map_constants.asm) :
        # le jeu lit les octets suivants de la ROM ; on complète avec le bloc de bordure.
        blk += bytes([objects["border_block"]]) * missing
        bst = (root / tilesets[info["tileset"]]["blockset"]).read_bytes()
        rows = []
        for sy in range(height * 2):
            row = []
            for sx in range(width * 2):
                block = blk[(sy // 2) * width + sx // 2]
                tile_y = (sy % 2) * SQUARE_TILES + 1          # tuile en bas à gauche
                tile_x = (sx % 2) * SQUARE_TILES
                row.append(bst[block * BLOCK_TILES * BLOCK_TILES + tile_y * BLOCK_TILES + tile_x])
            rows.append(bytes(row).hex())
        # Les constantes d'objet (object_const_def, à partir de 1) peuvent dépasser les
        # object_event (ex. SILPHCO7F_UNUSED, référencée par les objets activables).
        for index, obj_name in enumerate(objects.pop("object_names"), 1):
            object_index[(name, obj_name)] = index
        maps[name] = {
            "id": consts[name], "name": name, "label": label, "tileset": info["tileset"],
            "width": width, "height": height,
            "border_block": objects["border_block"],
            "connections": connections,
            "warps": objects["warps"], "signs": objects["signs"], "objects": objects["objects"],
            "gym_leader_no": gym_numbers.get(label),
            "padded_blocks": missing,
            "wild": None,
            "toggles": [],
            "tiles": rows,
        }

    for name, table in _wild(root, consts, map_names).items():
        if name in maps:
            maps[name]["wild"] = table
    toggles = _toggles(root, consts, object_index)
    for toggle_id, entry in enumerate(toggles):
        if entry["map"] not in maps:      # carte inutilisée (ex. UNUSED_MAP_F4)
            continue
        maps[entry["map"]]["toggles"].append({
            "toggle": toggle_id, "object": entry["object"], "initially_visible": entry["initially_visible"],
        })

    files = {
        f"maps/{name}.json": json.dumps(data, ensure_ascii=False, indent=1) + "\n"
        for name, data in sorted(maps.items())
    }
    files["tilesets.json"] = json.dumps({
        "source_commit": POKERED_COMMIT,
        "tilesets": tilesets,
        "rules": _movement_rules(root, consts, tileset_by_id),
        "toggles": toggles,
    }, ensure_ascii=False, indent=1) + "\n"
    return files


def main() -> int:
    args = base_parser(__doc__.splitlines()[1], OUT).parse_args()
    root = args.pokered or fetch_pokered(args.cache_dir)
    files = render(root)
    stale = []
    for rel, content in files.items():
        path = args.out / rel
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != content:
            stale.append(rel)
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
    extra = {p.relative_to(args.out).as_posix() for p in args.out.rglob("*.json")} - set(files)
    if args.check:
        for rel in stale + sorted(extra):
            print(f"PÉRIMÉ : {rel}")
        print(f"{len(files)} fichiers, {len(stale) + len(extra)} périmés")
        return 1 if stale or extra else 0
    for rel in extra:
        (args.out / rel).unlink()
    print(f"{len(files)} fichiers → {args.out.relative_to(ROOT)} ({len(stale)} modifiés)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
