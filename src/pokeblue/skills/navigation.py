"""NavigationSkill (baseline scriptée) : plus court chemin sur les grilles de la Phase 2.

Objectif (`Goal("navigate", …)`) :
    map       carte d'arrivée (ou `maps` : plusieurs cartes, la plus proche gagne)
    squares   cases visées dans cette carte (défaut : n'importe laquelle)
    object    nom d'objet (PNJ, panneau…) à aborder : on vise les cases voisines, ou la
              case située de l'autre côté d'un comptoir, en regardant l'objet
    face      direction à regarder une fois arrivé
    targets   variante : liste de [x, y, face] (une direction par case)
    interact  presser A une fois arrivé et orienté

Le chemin est recalculé quand le joueur n'est plus dessus (warp, combat, script). Une
case qui ne se libère pas (PNJ en travers) est contournée temporairement ; un arbre à
couper ou de l'eau sur le chemin est signalé dans `needs_field_move` (l'orchestrateur
lance alors le skill de capacité de terrain).
"""

from __future__ import annotations

from pokeblue.knowledge.gen1_data import MAPS
from pokeblue.knowledge.maps import load_map, world_rules
from pokeblue.knowledge.navigation import DIRECTIONS, Navigator, Path, Square
from pokeblue.skills.base import BaseSkill, Button, Goal
from pokeblue.state import ram_symbols as sym
from pokeblue.state.game_state import GameState
from pokeblue.state.mode_detector import Mode, detect_mode
from pokeblue.state.screen import Screen

FACING = {
    "down": sym.SPRITE_FACING_DOWN, "up": sym.SPRITE_FACING_UP,
    "left": sym.SPRITE_FACING_LEFT, "right": sym.SPRITE_FACING_RIGHT,
}
# Tuile de l'écran devant le joueur selon sa direction (_GetTileAndCoordsInFrontOfPlayer,
# engine/overworld/player_state.asm) : c'est celle que testent Coupe et les collisions.
TILE_IN_FRONT = {"down": (8, 11), "up": (8, 7), "left": (6, 9), "right": (10, 9)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
STUCK_LIMIT = 3          # pas sans bouger avant de contourner la case suivante
TEMP_BLOCK_STEPS = 40    # durée d'un contournement
WARP_STUCK_LIMIT = 2     # pressions sans effet sur un warp avant d'en sortir
NO_PATH_PATIENCE = 30    # replanifications sans chemin avant l'échec (PNJ de passage…)


def here_square(state: GameState) -> Square:
    return Square(MAPS[state.map_id].name, state.x, state.y)


def direction_between(a: Square, b: Square) -> str | None:
    """Direction d'un pas (ou d'un saut de corniche) dans une même carte."""
    dx, dy = b.x - a.x, b.y - a.y
    if dx and not dy:
        return "right" if dx > 0 else "left"
    if dy and not dx:
        return "down" if dy > 0 else "up"
    return None


def live_tile_ahead(state: GameState, direction: str) -> int:
    """Tuile réellement affichée sur la case voisine (un arbre coupé a disparu de
    l'écran mais pas des données de la carte)."""
    x, y = TILE_IN_FRONT[direction]
    return Screen(state.tilemap).tile(x, y)


def facing_squares(nav: Navigator, map_name: str, ox: int, oy: int) -> dict[Square, str]:
    """Cases d'où l'on peut parler à ce qui occupe (ox, oy), avec la direction à
    regarder : cases voisines, ou de l'autre côté d'un comptoir."""
    data = load_map(map_name)
    grid = nav.grid(map_name)
    counters = set(world_rules()["tilesets"][data.tileset]["counter_tiles"])
    targets = {}
    for name, (dx, dy) in DIRECTIONS.items():
        facing = OPPOSITE[name]
        x, y = ox + dx, oy + dy
        if data.in_bounds(x, y) and nav.enterable(grid, x, y):
            targets[Square(map_name, x, y)] = facing
        elif data.in_bounds(x, y) and grid.tile(x, y) in counters:
            x2, y2 = x + dx, y + dy                  # parler par-dessus un comptoir
            if data.in_bounds(x2, y2) and nav.enterable(grid, x2, y2):
                targets[Square(map_name, x2, y2)] = facing
    return targets


class NavigationSkill(BaseSkill):
    """Voir le docstring du module. `needs_field_move` : capacité de terrain demandée
    (l'orchestrateur la lance puis appelle `field_move_done`)."""

    name = "navigation"
    budget_steps = 1500

    def start(self, goal: Goal, state: GameState) -> None:
        super().start(goal, state)
        self.path: Path | None = None
        self.last_square: Square | None = None
        self.last_button: Button | None = None
        self.stuck = 0
        self.temp_blocked: dict[Square, int] = {}
        self.interacted = False
        self.needs_field_move: str | None = None
        self.targets: dict[Square, str | None] = {}
        self.no_path = 0

    # ── Cibles ────────────────────────────────────────────────────────────────

    def _compute_targets(self, state: GameState, nav: Navigator) -> dict[Square, str | None]:
        goal = self.goal
        if goal.get("maps"):
            return {sq: None for name in goal["maps"] for sq in self._all_squares(name)}
        target_map = goal["map"]
        if goal.get("object"):
            return self._object_targets(state, nav, target_map, goal["object"])
        if goal.get("targets"):
            return {Square(target_map, x, y): face for x, y, face in goal["targets"]}
        squares = goal.get("squares")
        face = goal.get("face")
        if squares:
            return {Square(target_map, x, y): face for x, y in squares}
        return dict.fromkeys(self._all_squares(target_map), face)

    @staticmethod
    def _all_squares(name: str) -> list[Square]:
        data = load_map(name)
        return [Square(name, x, y) for y in range(data.square_height) for x in range(data.square_width)]

    def _object_targets(self, state, nav, map_name, object_name) -> dict[Square, str | None]:
        data = load_map(map_name)
        obj = next((o for o in data.objects if o.name == object_name), None)
        if obj is None:
            self.fail(f"objet inconnu {object_name} dans {map_name}")
            return {}
        ox, oy = obj.x, obj.y
        if MAPS[state.map_id].name == map_name:       # position courante si la carte est chargée
            ox, oy = next(((x, y) for slot, x, y in state.sprites if slot == obj.index), (ox, oy))
        return facing_squares(nav, map_name, ox, oy)

    # ── Planification ─────────────────────────────────────────────────────────

    def _navigator(self, state: GameState) -> Navigator:
        nav = Navigator.from_state(state, self.goal.get("abilities"))
        for sq in self.temp_blocked:
            nav.grid(sq.map).blocked.add((sq.x, sq.y))
        return nav

    def _replan(self, state: GameState) -> None:
        nav = self._navigator(state)
        self.targets = self._compute_targets(state, nav)
        if not self.targets:
            self.path = None
            return
        self.path = nav.search([here_square(state)], lambda sq: sq in self.targets)
        self._nav = nav

    # ── Action ────────────────────────────────────────────────────────────────

    def _act(self, state: GameState) -> Button | None:
        if detect_mode(state) is not Mode.OVERWORLD or state.scripted:
            self.last_button = None           # scène scriptée : le jeu déplace le joueur
            return None
        here = here_square(state)
        if not load_map(here.map).in_bounds(here.x, here.y):
            return None                       # changement de carte en cours
        self._age_temp_blocks()
        if self.last_button in DIRECTIONS and here == self.last_square:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_square = here

        if self.stuck >= WARP_STUCK_LIMIT and self.path and here in self.path.squares:
            nxt = self.path.squares[self.path.squares.index(here) + 1]
            if nxt.map != here.map:
                # Un warp sur lequel on vient d'arriver ne se reprend qu'après en être
                # sorti (BIT_STANDING_ON_WARP) : un pas de côté, puis on revient.
                self.stuck = 0
                self.last_button = self._step_off(here)
                return self.last_button

        if self.path is None or here not in self.path.squares or self.stuck >= STUCK_LIMIT:
            if self.stuck >= STUCK_LIMIT:
                self._block_next(here)
                self.stuck = 0
            self._replan(state)
            if self.path is None:
                self.no_path += 1
                if self.no_path >= NO_PATH_PATIENCE and self.failure is None:
                    self.fail(f"aucun chemin vers {self.goal.get('map') or self.goal.get('maps')}")
                return None
            self.no_path = 0

        if here in self.targets:
            return self._arrive(state, self.targets[here])

        squares = self.path.squares
        nxt = squares[squares.index(here) + 1]
        button = self._step_button(state, here, nxt)
        self.last_button = button
        return button

    def _arrive(self, state: GameState, face: str | None) -> Button | None:
        if face and state.facing != FACING[face]:
            self.last_button = None          # se tourner ne déplace pas : pas un blocage
            return face
        if self.goal.get("interact") and not self.interacted:
            self.interacted = True
            return "a"
        self.done = True
        return None

    def _step_button(self, state: GameState, here: Square, nxt: Square) -> Button | None:
        if nxt.map == here.map:
            grid = self._nav.grid(here.map)
            ts = grid.data.tileset
            tile = grid.tile(nxt.x, nxt.y)
            direction = direction_between(here, nxt)
            if abs(nxt.x - here.x) + abs(nxt.y - here.y) == 1:
                if tile in self._nav._cut.get(ts, ()) and tile not in self._nav._passable[ts] \
                        and live_tile_ahead(state, direction) == tile:   # pas déjà coupé
                    return self._field_move(state, "CUT", direction)
                here_tile = grid.tile(here.x, here.y)
                if self._nav._is_water(ts, tile) and not self._nav._is_water(ts, here_tile) \
                        and state.walk_bike_surf != 2:
                    return self._field_move(state, "SURF", direction)
            return direction
        data = load_map(here.map)
        if data.warp_at(here.x, here.y):
            if here.y == data.square_height - 1:
                return "down"
            if here.y == 0:
                return "up"
            if here.x == 0:
                return "left"
            if here.x == data.square_width - 1:
                return "right"
            return self.last_button if self.last_button in DIRECTIONS else "down"
        if here.y == 0:
            return "up"
        if here.y == data.square_height - 1:
            return "down"
        return "left" if here.x == 0 else "right"

    def field_move_done(self) -> None:
        self.needs_field_move = None
        self.path = None                         # replanifier : l'obstacle a disparu

    def _step_off(self, here: Square) -> Button | None:
        for sq in self._nav.neighbors(here):
            if sq.map == here.map:
                return direction_between(here, sq)
        return None

    def _field_move(self, state: GameState, move: str, direction: str) -> Button | None:
        if state.facing != FACING[direction]:
            return direction                     # d'abord faire face à l'obstacle
        self.needs_field_move = move
        return None

    def _block_next(self, here: Square) -> None:
        if self.path and here in self.path.squares:
            i = self.path.squares.index(here)
            if i + 1 < len(self.path.squares) and self.path.squares[i + 1].map == here.map:
                self.temp_blocked[self.path.squares[i + 1]] = TEMP_BLOCK_STEPS

    def _age_temp_blocks(self) -> None:
        for sq in list(self.temp_blocked):
            self.temp_blocked[sq] -= 1
            if self.temp_blocked[sq] <= 0:
                del self.temp_blocked[sq]
                self.path = None


class Navigating:
    """Pour un skill composite qui se déplace avec un NavigationSkill `self.nav` :
    relaie ses demandes de capacité de terrain à l'orchestrateur."""

    nav: NavigationSkill

    @property
    def needs_field_move(self) -> str | None:
        return getattr(self, "nav", None) and self.nav.needs_field_move

    def field_move_done(self) -> None:
        self.nav.field_move_done()
