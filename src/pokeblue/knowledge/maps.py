"""Cartes du jeu : chargement des données générées par `scripts/gen_maps.py`.

Coordonnées en *cases* de déplacement (2×2 tuiles), comme wXCoord / wYCoord. Une
carte de `width` × `height` blocs fait donc 2·width × 2·height cases. `tile(x, y)`
renvoie la tuile représentative de la case (en bas à gauche), celle que le moteur
teste pour les collisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
LAST_MAP = "LAST_MAP"   # destination de warp : la carte d'où l'on vient (wLastMap)


@dataclass(frozen=True, slots=True)
class Warp:
    x: int
    y: int
    map: str          # carte d'arrivée, ou LAST_MAP
    warp: int         # n° du warp d'arrivée dans cette carte, à partir de 1


@dataclass(frozen=True, slots=True)
class Connection:
    direction: str    # north / south / west / east
    map: str
    offset: int       # décalage en blocs de la carte voisine (voir connection, macros/scripts/maps.asm)


@dataclass(frozen=True, slots=True)
class Sign:
    x: int
    y: int
    text: str


@dataclass(frozen=True, slots=True)
class MapObject:
    index: int                        # n° d'objet (à partir de 1), wSpriteStateData
    name: str | None                  # constante pokered (ex. ROUTE12_SNORLAX)
    x: int
    y: int
    sprite: str
    movement: str                     # STAY / WALK
    range_or_direction: str
    text: str
    trainer: tuple[str, int] | None   # (classe, n° d'équipe à partir de 1)
    item: str | None                  # objet ramassable


@dataclass(frozen=True, slots=True)
class Toggle:
    toggle: int               # bit de wToggleableObjectFlags (1 = objet masqué)
    object: int               # n° d'objet sur la carte
    initially_visible: bool


@dataclass(frozen=True, slots=True)
class WildTable:
    grass_rate: int
    grass: tuple[tuple[int, str], ...]   # (niveau, espèce) ×10
    water_rate: int
    water: tuple[tuple[int, str], ...]


@dataclass(frozen=True, slots=True)
class MapData:
    id: int
    name: str
    label: str
    tileset: str
    width: int                # en blocs
    height: int
    border_block: int
    connections: tuple[Connection, ...]
    warps: tuple[Warp, ...]
    signs: tuple[Sign, ...]
    objects: tuple[MapObject, ...]
    toggles: tuple[Toggle, ...]
    wild: WildTable | None
    gym_leader_no: int | None
    blocks: bytes             # n° de bloc, width × height (fichier .blk)
    tiles: tuple[bytes, ...]  # une ligne par rangée de cases

    @property
    def square_width(self) -> int:
        return self.width * 2

    @property
    def square_height(self) -> int:
        return self.height * 2

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.square_width and 0 <= y < self.square_height

    def tile(self, x: int, y: int) -> int:
        return self.tiles[y][x]

    def warp_at(self, x: int, y: int) -> Warp | None:
        return next((w for w in self.warps if (w.x, w.y) == (x, y)), None)

    def connection(self, direction: str) -> Connection | None:
        return next((c for c in self.connections if c.direction == direction), None)

    def trainers(self) -> tuple[MapObject, ...]:
        return tuple(o for o in self.objects if o.trainer)


def _from_json(raw: dict) -> MapData:
    wild = raw["wild"]
    return MapData(
        id=raw["id"], name=raw["name"], label=raw["label"], tileset=raw["tileset"],
        width=raw["width"], height=raw["height"], border_block=raw["border_block"],
        connections=tuple(Connection(**c) for c in raw["connections"]),
        warps=tuple(Warp(**w) for w in raw["warps"]),
        signs=tuple(Sign(**s) for s in raw["signs"]),
        objects=tuple(MapObject(
            index=o["index"], name=o["name"], x=o["x"], y=o["y"], sprite=o["sprite"],
            movement=o["movement"], range_or_direction=o["range_or_direction"], text=o["text"],
            trainer=(o["trainer"]["class"], o["trainer"]["party"]) if o["trainer"] else None,
            item=o["item"],
        ) for o in raw["objects"]),
        toggles=tuple(Toggle(**t) for t in raw["toggles"]),
        wild=WildTable(
            grass_rate=wild["grass_rate"], grass=tuple(map(tuple, wild["grass"])),
            water_rate=wild["water_rate"], water=tuple(map(tuple, wild["water"])),
        ) if wild else None,
        gym_leader_no=raw["gym_leader_no"],
        blocks=bytes.fromhex(raw["blocks"]),
        tiles=tuple(bytes.fromhex(row) for row in raw["tiles"]),
    )


@cache
def map_names() -> tuple[str, ...]:
    """Noms des cartes qui ont des données (les cartes inutilisées n'en ont pas)."""
    return tuple(sorted(p.stem for p in (DATA_DIR / "maps").glob("*.json")))


@cache
def load_map(name: str) -> MapData:
    path = DATA_DIR / "maps" / f"{name}.json"
    if not path.exists():
        raise KeyError(f"carte sans données : {name}")
    return _from_json(json.loads(path.read_text(encoding="utf-8")))


@cache
def map_by_id(map_id: int) -> MapData:
    from pokeblue.knowledge.gen1_data import MAPS
    return load_map(MAPS[map_id].name)


@cache
def world_rules() -> dict:
    """Tilesets (tuiles praticables, herbe…) et règles de déplacement (tilesets.json)."""
    return json.loads((DATA_DIR / "tilesets.json").read_text(encoding="utf-8"))
