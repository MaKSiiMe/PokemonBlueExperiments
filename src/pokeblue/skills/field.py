"""Skills composites hors combat : soin au Centre Pokémon, entraînement dans l'herbe.

Ils s'appuient sur NavigationSkill pour les déplacements et sur la politique de
dialogue pour les textes (`handles_menus = True` : l'orchestrateur leur laisse les
menus et dialogues qu'ils provoquent).
"""

from __future__ import annotations

from functools import cache

from pokeblue.knowledge.gen1_data import MAPS
from pokeblue.knowledge.maps import load_map, map_names, world_rules
from pokeblue.knowledge.navigation import DIRECTIONS
from pokeblue.skills.base import BaseSkill, Button, Goal, SkillStatus
from pokeblue.skills.dialog import dialog_button
from pokeblue.skills.navigation import Navigating, NavigationSkill, here_square
from pokeblue.state.game_state import GameState
from pokeblue.state.mode_detector import Mode, detect_mode

NURSE_SPRITE = "SPRITE_NURSE"


@cache
def nurse_maps() -> dict[str, str]:
    """Cartes avec une infirmière → nom de l'objet infirmière."""
    out = {}
    for name in map_names():
        nurse = next((o for o in load_map(name).objects if o.sprite == NURSE_SPRITE), None)
        if nurse is not None:
            out[name] = nurse.name
    return out


def fully_healed(state: GameState) -> bool:
    return all(mon.hp == mon.max_hp and not mon.status for mon in state.party) and all(
        pp > 0 for mon in state.party for move, pp in zip(mon.moves, mon.pp, strict=True) if move)


class HealSkill(Navigating, BaseSkill):
    """Aller au Centre Pokémon le plus proche et faire soigner l'équipe."""

    name = "heal"
    budget_steps = 1500
    handles_menus = True

    def start(self, goal: Goal, state: GameState) -> None:
        super().start(goal, state)
        self.nav = NavigationSkill()
        self.nav_goal: Goal | None = None
        self.talked = False

    def _navigate(self, goal: Goal, state: GameState) -> Button | None:
        if self.nav_goal != goal:
            self.nav_goal = goal
            self.nav.start(goal, state)
        button = self.nav.act(state)
        if self.nav.status(state) in (SkillStatus.FAILURE, SkillStatus.TIMEOUT):
            self.fail(f"navigation : {self.nav.failure or 'timeout'}")
        return button

    def _act(self, state: GameState) -> Button | None:
        mode = detect_mode(state)
        if mode is not Mode.OVERWORLD:
            return dialog_button(state)       # « Shall we heal your POKéMON? » → OUI
        if self.talked:
            self.done = True
            return None
        here = MAPS[state.map_id].name
        if here not in nurse_maps():
            return self._navigate(Goal("navigate", {"maps": sorted(nurse_maps())}), state)
        button = self._navigate(Goal("navigate", {"map": here, "object": nurse_maps()[here],
                                                   "interact": True}), state)
        if self.nav.interacted:
            self.talked = True
        return button

    def _succeeded(self, state: GameState) -> bool:
        return self.talked and fully_healed(state) and detect_mode(state) is Mode.OVERWORLD


def grass_squares(map_name: str) -> list[tuple[int, int]]:
    data = load_map(map_name)
    grass = world_rules()["tilesets"][data.tileset]["grass_tile"]
    if grass is None or data.wild is None or not data.wild.grass_rate:
        return []
    return [(x, y) for y in range(data.square_height) for x in range(data.square_width)
            if data.tile(x, y) == grass]


class TrainSkill(Navigating, BaseSkill):
    """Arpenter l'herbe d'une carte jusqu'à ce que l'équipe atteigne `until_level`.

    Objectif : map, until_level, min_hp (fraction de PV de l'équipe sous laquelle on
    rend la main pour aller se soigner). Les combats sont menés par BattleSkill.
    """

    name = "train"
    budget_steps = 6000
    handles_menus = False

    def start(self, goal: Goal, state: GameState) -> None:
        super().start(goal, state)
        self.nav = NavigationSkill()
        self.nav_started = False
        self.squares = set(grass_squares(goal["map"]))
        if not self.squares:
            self.fail(f"pas d'herbe sur {goal['map']}")
        self.toggle = 0

    def _act(self, state: GameState) -> Button | None:
        if detect_mode(state) is not Mode.OVERWORLD or state.scripted:
            return None
        here = here_square(state)
        if here.map == self.goal["map"] and (here.x, here.y) in self.squares:
            self.nav_started = False
            return self._pace(here)
        if not self.nav_started:
            self.nav.start(Goal("navigate", {"map": self.goal["map"], "squares": sorted(self.squares)}), state)
            self.nav_started = True
        button = self.nav.act(state)
        if self.nav.status(state) is SkillStatus.FAILURE:
            self.fail(f"navigation : {self.nav.failure}")
        elif self.nav.status(state) is not SkillStatus.RUNNING:
            self.nav_started = False
        return button

    def _pace(self, here) -> Button:
        """Aller-retour entre deux cases d'herbe voisines."""
        self.toggle ^= 1
        options = [d for d, (dx, dy) in DIRECTIONS.items() if (here.x + dx, here.y + dy) in self.squares]
        if not options:
            return "up"
        return options[self.toggle % len(options)]

    def _succeeded(self, state: GameState) -> bool:
        if state.battle is not None:
            return False
        if max((m.level for m in state.party), default=0) >= self.goal["until_level"]:
            return True
        return state.party_hp < self.goal.get("min_hp", 0.4) * state.party_max_hp
