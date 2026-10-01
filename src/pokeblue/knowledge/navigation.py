"""Navigation entre cartes : graphe de cases, contraintes de progression, `path()`.

Un nœud est une case (carte, x, y). Les transitions reproduisent les règles du
moteur relevées dans pokered (voir tilesets.json et docs/knowledge.md) :

- marche : tuile praticable du tileset (coll_tiles), sans paire de tuiles interdite
  (dénivelés, TilePairCollisionsLand) ;
- corniche : saut d'une case dans le sens de la corniche (LedgeTiles, tileset
  OVERWORLD uniquement), à sens unique ;
- eau : tuiles d'eau, si la capacité SURF est disponible ; débarquement sur une
  tuile praticable ;
- arbre : tuile d'arbre coupable, si CUT est disponible ;
- warps : d'une case de warp à la case du warp d'arrivée (LAST_MAP : vers chaque
  carte qui mène à celle-ci) ; connexions : passage au bord de la carte voisine.

Obstacles dépendant de la progression (`Progress`) : PNJ immobiles visibles (objets
activables masqués ou non), rochers (franchissables avec STRENGTH), verrous scriptés
(gates.yaml) et blocs remplacés selon un événement (block_events.yaml).

Simplifications assumées : les PNJ qui se déplacent sont ignorés, un rocher est
considéré comme franchissable dès que Force est disponible (les énigmes de rochers
relèvent de la Phase 5), et les dalles tournantes, trous et courants ne sont pas
modélisés.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import IntFlag
from functools import cache
from pathlib import Path

import yaml

from pokeblue.knowledge.gen1_data import EVENT_IDS, ITEM_IDS, MOVE_IDS
from pokeblue.knowledge.maps import LAST_MAP, MapData, load_map, map_names, world_rules
from pokeblue.state import ram_symbols as sym

KNOWLEDGE_DIR = Path(__file__).parent
DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
EDGE_OF = {"north": "up", "south": "down", "west": "left", "east": "right"}
BOULDER_SPRITE = "SPRITE_BOULDER"


class Ability(IntFlag):
    """Capacités de terrain qui changent les déplacements possibles."""

    NONE = 0
    CUT = 1
    SURF = 2
    STRENGTH = 4


# Capacité → (attaque, badge exigé hors combat) : engine/menus/start_sub_menus.asm.
FIELD_MOVES = {
    Ability.CUT: ("CUT", "CASCADEBADGE"),
    Ability.SURF: ("SURF", "SOULBADGE"),
    Ability.STRENGTH: ("STRENGTH", "RAINBOWBADGE"),
}


def _badge_bit(name: str) -> int:
    return getattr(sym, f"BIT_{name}")


@dataclass(frozen=True)
class Progress:
    """Ce qui est débloqué dans la partie.

    `events = None` signifie « tous les drapeaux actifs » et `hidden_objects = None`
    « état initial des objets activables » ; `Progress.everything()` combine tous les
    déblocages (planification d'un trajet en fin de jeu).
    """

    events: int | None = 0            # bitset des drapeaux d'événement
    badges: int = 0
    items: frozenset[int] = frozenset()
    money: int = 0
    status_flags1: int = 0
    hidden_objects: int | None = None
    abilities: Ability = Ability.NONE

    @classmethod
    def from_state(cls, state, abilities: Ability | None = None) -> Progress:
        """Progression lue dans un GameState ; capacités déduites de l'équipe et des badges."""
        if abilities is None:
            known = {move for mon in state.party for move in mon.moves}
            abilities = Ability.NONE
            for ability, (move, badge) in FIELD_MOVES.items():
                if MOVE_IDS[move] in known and state.has_badge(_badge_bit(badge)):
                    abilities |= ability
        return cls(
            events=state.event_flags, badges=state.badges,
            items=frozenset(item for item, qty in state.bag if qty),
            money=state.money, status_flags1=state.status_flags1,
            hidden_objects=state.hidden_objects, abilities=abilities,
        )

    @classmethod
    def everything(cls) -> Progress:
        return cls(events=None, badges=0xFF, items=frozenset(ITEM_IDS.values()),
                   money=999_999, status_flags1=0xFF, hidden_objects=-1,
                   abilities=Ability.CUT | Ability.SURF | Ability.STRENGTH)

    def event(self, event: int) -> bool:
        return True if self.events is None else bool(self.events >> event & 1)

    def object_hidden(self, toggle: int, initially_visible: bool) -> bool:
        if self.hidden_objects is None:
            return not initially_visible
        return bool(self.hidden_objects >> toggle & 1)

    def satisfies(self, condition: dict) -> bool:
        """Condition au format de gates.yaml (toutes les clauses doivent être vraies)."""
        checks = [
            *(self.event(EVENT_IDS[e]) for e in condition.get("events", ())),
            *(not self.event(EVENT_IDS[e]) or self.events is None
              for e in condition.get("events_unset", ())),
            *(bool(self.badges >> _badge_bit(b) & 1) for b in condition.get("badges", ())),
            *(ITEM_IDS[i] in self.items for i in condition.get("items", ())),
            *(bool(self.status_flags1 >> getattr(sym, f) & 1)
              for f in condition.get("status_flags1", ())),
            self.money >= condition.get("money", 0),
        ]
        return all(checks)


# ── Données de connaissance ───────────────────────────────────────────────────

@cache
def gates() -> tuple[dict, ...]:
    return tuple(yaml.safe_load((KNOWLEDGE_DIR / "gates.yaml").read_text(encoding="utf-8"))["gates"])


@cache
def block_events() -> tuple[dict, ...]:
    data = yaml.safe_load((KNOWLEDGE_DIR / "block_events.yaml").read_text(encoding="utf-8"))
    return tuple(data["block_events"])


def gate_squares(gate: dict) -> list[tuple[int, int]]:
    squares = [tuple(sq) for sq in gate.get("squares", ())]
    if "rows" in gate:
        width = load_map(gate["map"]).square_width
        for row in gate["rows"]:
            last = min(row.get("x_max", width - 1), width - 1)
            squares += [(x, row["y"]) for x in range(last + 1)]
    return squares


def gate_open(gate: dict, progress: Progress) -> bool:
    if "open_when_any" in gate:
        return any(progress.satisfies(c) for c in gate["open_when_any"])
    return progress.satisfies(gate["open_when"])


@cache
def _warp_sources() -> dict[str, list[str]]:
    """Carte → cartes dont un warp y mène (résolution de LAST_MAP)."""
    sources: dict[str, list[str]] = {}
    for name in map_names():
        for warp in load_map(name).warps:
            if warp.map != LAST_MAP and name not in sources.setdefault(warp.map, []):
                sources[warp.map].append(name)
    return sources


@cache
def _last_map_candidates(name: str, at: tuple[int, int]) -> tuple[str, ...]:
    """Cartes d'où l'on a pu entrer par la porte située en `at`.

    LAST_MAP renvoie à wLastMap, la carte d'où l'on vient. Une porte (une ou deux
    cases de warp voisines) n'est empruntée que depuis les cartes dont un warp arrive
    sur elle : c'est ainsi qu'une porte de Route 22 mène à la Route 22 d'un côté et à
    la Route 23 de l'autre (le script Route22Gate règle wLastMap selon le côté).
    """
    warps = load_map(name).warps
    matches = []
    for source in _warp_sources().get(name, []):
        for sw in load_map(source).warps:
            if sw.map == name and 1 <= sw.warp <= len(warps):
                arrival = warps[sw.warp - 1]
                if abs(arrival.x - at[0]) + abs(arrival.y - at[1]) <= 1:
                    matches.append(source)
                    break
    return tuple(matches) or tuple(_warp_sources().get(name, []))


# ── Grille d'une carte pour une progression donnée ────────────────────────────

@dataclass
class MapGrid:
    data: MapData
    tiles: list[bytearray]
    blocked: set[tuple[int, int]] = field(default_factory=set)
    boulders: set[tuple[int, int]] = field(default_factory=set)

    @classmethod
    def build(cls, name: str, progress: Progress,
              live: dict[int, tuple[int, int]] | None = None) -> MapGrid:
        """`live` : positions courantes (n° d'objet → case) des PNJ de la carte, lues en
        RAM ; elles remplacent les positions initiales (un dresseur qui s'est avancé
        reste là où il s'est arrêté)."""
        data = load_map(name)
        tiles = [bytearray(row) for row in data.tiles]
        squares_of = bytes.fromhex(world_rules()["tilesets"][data.tileset]["block_squares"])
        for ev in block_events():
            if ev["map"] != name:
                continue
            active = progress.event(EVENT_IDS[ev["event"]])
            solvable = ev.get("solvable_with")
            if solvable and progress.abilities & Ability[solvable]:
                active = True   # énigme résoluble : le joueur peut activer l'événement
            if active != ev["when_set"]:
                continue
            bx, by = ev["block"]
            for i in range(4):
                tiles[2 * by + i // 2][2 * bx + i % 2] = squares_of[ev["block_id"] * 4 + i]
        grid = cls(data, tiles)
        toggles = {t.object: t for t in data.toggles}
        for obj in data.objects:
            toggle = toggles.get(obj.index)
            if toggle and progress.object_hidden(toggle.toggle, toggle.initially_visible):
                continue
            if live is not None:
                if obj.index in live:
                    target = grid.boulders if obj.sprite == BOULDER_SPRITE else grid.blocked
                    target.add(live[obj.index])
                continue
            if obj.sprite == BOULDER_SPRITE:
                grid.boulders.add((obj.x, obj.y))
            elif obj.movement == "STAY":
                grid.blocked.add((obj.x, obj.y))
        for gate in gates():
            if gate["map"] == name and not gate_open(gate, progress):
                grid.blocked.update(gate_squares(gate))
        return grid

    def tile(self, x: int, y: int) -> int:
        return self.tiles[y][x]


@dataclass(frozen=True, slots=True)
class Square:
    map: str
    x: int
    y: int


@dataclass(frozen=True)
class Path:
    squares: tuple[Square, ...]

    @property
    def maps(self) -> tuple[str, ...]:
        out: list[str] = []
        for sq in self.squares:
            if not out or out[-1] != sq.map:
                out.append(sq.map)
        return tuple(out)

    def __len__(self) -> int:
        return len(self.squares)


class Navigator:
    """Graphe de cases pour une progression donnée (grilles construites à la demande)."""

    def __init__(self, progress: Progress,
                 live_sprites: dict[str, dict[int, tuple[int, int]]] | None = None) -> None:
        self.progress = progress
        self._live = live_sprites or {}
        self._grids: dict[str, MapGrid] = {}
        rules = world_rules()["rules"]
        self._water_tilesets = set(rules["water_tilesets"])
        water = rules["water_tiles"]
        self._water = {
            ts: {t for t in water["tiles"] if ts not in water["not_in_tilesets"].get(f"{t:#04x}", [])}
            for ts in self._water_tilesets
        }
        self._cut = {ts: set(tiles) for ts, tiles in rules["cut_tree_tiles"].items()
                     if isinstance(tiles, list)}
        self._ledges = {(DIRECTIONS[le["direction"]], le["from_tile"], le["ledge_tile"])
                        for le in rules["ledges"]}
        self._ledge_tilesets = set(rules["ledges_only_in"])
        self._pairs = {kind: {(p["tileset"], frozenset(p["tiles"])) for p in pairs}
                       for kind, pairs in rules["tile_pair_collisions"].items()}
        self._passable = {ts: set(info["passable"]) for ts, info in world_rules()["tilesets"].items()}

    @classmethod
    def from_state(cls, state, abilities: Ability | None = None) -> Navigator:
        """Navigateur pour un GameState : progression et PNJ de la carte courante."""
        from pokeblue.knowledge.gen1_data import MAPS
        live = {slot: (x, y) for slot, x, y in state.sprites}
        return cls(Progress.from_state(state, abilities), {MAPS[state.map_id].name: live})

    def grid(self, name: str) -> MapGrid:
        if name not in self._grids:
            self._grids[name] = MapGrid.build(name, self.progress, self._live.get(name))
        return self._grids[name]

    # ── Règles d'une case ─────────────────────────────────────────────────────

    def _is_water(self, ts: str, tile: int) -> bool:
        return tile in self._water.get(ts, ())

    def enterable(self, grid: MapGrid, x: int, y: int) -> bool:
        """Case où l'on peut se trouver (à pied ou en surfant)."""
        if (x, y) in grid.blocked:
            return False
        if (x, y) in grid.boulders and not self.progress.abilities & Ability.STRENGTH:
            return False
        ts, tile = grid.data.tileset, grid.tile(x, y)
        if tile in self._passable[ts]:
            return True
        if self._is_water(ts, tile):
            return bool(self.progress.abilities & Ability.SURF)
        return tile in self._cut.get(ts, ()) and bool(self.progress.abilities & Ability.CUT)

    def _pair_blocked(self, ts: str, a: int, b: int, on_water: bool) -> bool:
        return (ts, frozenset((a, b))) in self._pairs["water" if on_water else "land"]

    def neighbors(self, sq: Square) -> Iterator[Square]:
        grid = self.grid(sq.map)
        data, ts = grid.data, grid.data.tileset
        here = grid.tile(sq.x, sq.y)
        on_water = self._is_water(ts, here)
        for name, (dx, dy) in DIRECTIONS.items():
            nx, ny = sq.x + dx, sq.y + dy
            if not data.in_bounds(nx, ny):
                target = self._across_connection(data, sq, name)
                if target is not None:
                    yield target
                continue
            there = grid.tile(nx, ny)
            if ts in self._ledge_tilesets and ((dx, dy), here, there) in self._ledges:
                lx, ly = nx + dx, ny + dy
                if data.in_bounds(lx, ly) and self.enterable(grid, lx, ly):
                    yield Square(sq.map, lx, ly)
                continue
            if self._pair_blocked(ts, here, there, on_water):
                continue
            if self.enterable(grid, nx, ny):
                yield Square(sq.map, nx, ny)
        warp = data.warp_at(sq.x, sq.y)
        if warp is not None:
            yield from self._warp_targets(sq.map, warp.map, warp.warp, (sq.x, sq.y))

    def _across_connection(self, data: MapData, sq: Square, direction: str) -> Square | None:
        edge = next((e for e, d in EDGE_OF.items() if d == direction), None)
        conn = data.connection(edge)
        if conn is None or conn.map not in map_names():
            return None
        target = load_map(conn.map)
        shift = -2 * conn.offset
        if edge == "north":
            x, y = sq.x + shift, target.square_height - 1
        elif edge == "south":
            x, y = sq.x + shift, 0
        elif edge == "west":
            x, y = target.square_width - 1, sq.y + shift
        else:
            x, y = 0, sq.y + shift
        if not target.in_bounds(x, y) or not self.enterable(self.grid(conn.map), x, y):
            return None
        return Square(conn.map, x, y)

    def _warp_targets(self, source: str, dest: str, warp_no: int,
                      at: tuple[int, int]) -> Iterator[Square]:
        candidates = _last_map_candidates(source, at) if dest == LAST_MAP else [dest]
        for name in candidates:
            if name not in map_names():
                continue
            warps = load_map(name).warps
            if 1 <= warp_no <= len(warps):
                w = warps[warp_no - 1]
                yield Square(name, w.x, w.y)

    # ── Recherche ─────────────────────────────────────────────────────────────

    def entry_squares(self, name: str) -> list[Square]:
        """Points d'entrée d'une carte : ses warps et les cases de bord reliées."""
        grid = self.grid(name)
        data = grid.data
        squares = [Square(name, w.x, w.y) for w in data.warps if self.enterable(grid, w.x, w.y)]
        for conn in data.connections:
            if conn.direction in ("north", "south"):
                y = 0 if conn.direction == "north" else data.square_height - 1
                edge = [(x, y) for x in range(data.square_width)]
            else:
                x = 0 if conn.direction == "west" else data.square_width - 1
                edge = [(x, y) for y in range(data.square_height)]
            squares += [Square(name, x, y) for x, y in edge if self.enterable(grid, x, y)]
        return squares

    def search(self, starts: Iterable[Square], goal) -> Path | None:
        """Plus court chemin (en cases) d'un départ vers une case satisfaisant `goal`."""
        parents: dict[Square, Square | None] = {}
        queue: deque[Square] = deque()
        for sq in starts:
            if sq not in parents:
                parents[sq] = None
                queue.append(sq)
        while queue:
            sq = queue.popleft()
            if goal(sq):
                out = [sq]
                while parents[out[-1]] is not None:
                    out.append(parents[out[-1]])
                return Path(tuple(reversed(out)))
            for nxt in self.neighbors(sq):
                if nxt not in parents:
                    parents[nxt] = sq
                    queue.append(nxt)
        return None


def path(map_a: str, map_b: str, progress: Progress | None = None,
         start: tuple[int, int] | None = None) -> Path | None:
    """Chemin de `map_a` (position `start`, ou ses points d'entrée) à n'importe quelle
    case de `map_b`, compte tenu de la progression (par défaut : jeu terminé)."""
    nav = Navigator(progress or Progress.everything())
    starts = [Square(map_a, *start)] if start else nav.entry_squares(map_a)
    return nav.search(starts, lambda sq: sq.map == map_b)
