"""StrategySkill (baseline à règles) : quel objectif poursuivre maintenant ?

Règles, dans l'ordre :
1. équipe vide → jalon courant (le starter) ;
2. PV de l'équipe sous `heal_below`, Pokémon K.O. ou plus de PP offensifs → soin ;
3. CS dans le sac, badge obtenu, attaque inconnue de l'équipe → l'apprendre ;
   moins de `min_potions` potions et assez d'argent → boutique la plus proche ;
4. niveau maximal de l'équipe sous le niveau conseillé du jalon → entraînement dans
   l'herbe atteignable la plus proche ;
5. sinon → objectif du jalon courant (`approach` / `steps` de progression.yaml, ou
   règle générique : aller sur la carte cible et parler au dresseur ciblé).

La décision est une fonction de l'état : l'orchestrateur la réévalue à chaque point
de décision (fin de skill, retour dans l'overworld après une scène ou un combat).
"""

from __future__ import annotations

from dataclasses import dataclass

from pokeblue.knowledge.gen1_data import EVENT_IDS, ITEM_IDS, MAPS, MOVE_IDS, is_damaging
from pokeblue.knowledge.maps import load_map, map_names
from pokeblue.knowledge.navigation import FIELD_MOVES, Navigator, Progress, Square
from pokeblue.knowledge.progression import (
    Milestone,
    next_milestone,
    required_level,
    rival_starter_name,
)
from pokeblue.skills.base import Goal
from pokeblue.skills.battle import HEALING_ITEMS
from pokeblue.skills.field import grass_squares, nurse_maps
from pokeblue.state import ram_symbols as sym
from pokeblue.state.game_state import GameState


@dataclass(frozen=True)
class StrategyConfig:
    heal_below: float = 0.5          # fraction des PV de l'équipe
    train_margin: int = 0            # niveaux au-delà du niveau conseillé
    battle_heal_below: float = 0.25  # Potion en combat sous cette fraction des PV
    flee_below: float = 0.25         # fuite d'un combat sauvage sous cette fraction des PV
    flee_when_ready: bool = True     # fuir les combats sauvages une fois le niveau atteint
    min_potions: int = 3             # en dessous : passer en boutique…
    buy_potions: int = 5             # … en acheter autant…
    shop_money: int = 1500           # … si l'on a au moins cet argent


def max_level(state: GameState) -> int:
    return max((mon.level for mon in state.party), default=0)


def needs_heal(state: GameState, threshold: float) -> bool:
    if not state.party:
        return False
    if any(mon.hp == 0 for mon in state.party):
        return True
    if state.party_hp < threshold * state.party_max_hp:
        return True
    lead = next(mon for mon in state.party if mon.hp > 0)
    return not any(pp for move, pp in zip(lead.moves, lead.pp, strict=True)
                   if move and is_damaging(move))


def _here(state: GameState) -> str:
    return MAPS[state.map_id].name


def hm_to_teach(state: GameState) -> Goal | None:
    """CS du sac utilisable hors combat (badge obtenu) qu'aucun Pokémon ne connaît.

    Simplification : la CS est apprise au premier Pokémon de l'équipe (les
    compatibilités CT/CS ne sont pas encore extraites de pokered)."""
    for move, badge in FIELD_MOVES.values():
        item = ITEM_IDS[f"HM_{move}"]
        known = any(MOVE_IDS[move] in mon.moves for mon in state.party)
        if state.item_count(item) and not known and state.has_badge(getattr(sym, f"BIT_{badge}")):
            return Goal("teach", {"item": f"HM_{move}", "move": move, "party_index": 0})
    return None


# ── Objectif d'un jalon ───────────────────────────────────────────────────────

def trainer_object(m: Milestone) -> str | None:
    """Objet du dresseur ciblé par le jalon sur sa carte (classe et n° d'équipe)."""
    if m.trainer_class is None:
        return None
    for obj in load_map(m.target_map).objects:
        if obj.trainer and obj.trainer[0] == m.trainer_class and (
                m.trainer_party is None or obj.trainer[1] == m.trainer_party):
            return obj.name
    return None


def milestone_goal(state: GameState, m: Milestone) -> Goal:
    """`approach` du jalon (progression.yaml), sinon : parler au dresseur ciblé, ou à
    défaut entrer dans la carte cible."""
    approach = m.current_approach(Progress.from_state(state))
    if approach and "skill" in approach:
        return Goal(approach["skill"], {k: v for k, v in approach.items() if k != "skill"})
    if approach:
        return Goal("navigate", {"map": m.target_map, **approach})
    obj = trainer_object(m)
    if obj:
        return Goal("navigate", {"map": m.target_map, "object": obj, "interact": True})
    return Goal("navigate", {"map": m.target_map})


# ── Stratégie ─────────────────────────────────────────────────────────────────

class Strategy:
    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    def milestone(self, state: GameState) -> Milestone | None:
        return next_milestone(state)

    def target_level(self, state: GameState, m: Milestone) -> int:
        return required_level(m, rival_starter_name(state)) + self.config.train_margin

    def decide(self, state: GameState) -> Goal:
        m = self.milestone(state)
        if m is None:
            return Goal("done")
        if not state.party:
            return self.milestone_goal(state, m)
        if needs_heal(state, self.config.heal_below) and self._center_reachable(state):
            return Goal("heal")
        teach = hm_to_teach(state)
        if teach:
            return teach
        shopping = self.shopping_goal(state)
        if shopping:
            return shopping
        level = self.target_level(state, m)
        if max_level(state) < level:
            grass_map = self.training_map(state)
            if grass_map:
                return Goal("train", {"map": grass_map, "until_level": level,
                                      "min_hp": self.config.heal_below, "milestone": m.id})
        return self.milestone_goal(state, m)

    def milestone_goal(self, state: GameState, m: Milestone) -> Goal:
        goal = milestone_goal(state, m)
        return Goal(goal.kind, {**goal.params, "milestone": m.id})

    def battle_goal(self, state: GameState) -> Goal:
        m = self.milestone(state)
        ready = m is None or max_level(state) >= self.target_level(state, m)
        return Goal("battle", {"flee_wild": self.config.flee_when_ready and ready,
                               "heal_below": self.config.battle_heal_below,
                               "flee_below": self.config.flee_below})

    def shopping_goal(self, state: GameState) -> Goal | None:
        """Racheter des potions (la meilleure vendue par la boutique la plus proche)."""
        potions = sum(state.item_count(ITEM_IDS[i]) for i in HEALING_ITEMS)
        if potions >= self.config.min_potions or state.money < self.config.shop_money:
            return None
        if not state.flag(EVENT_IDS["EVENT_OAK_GOT_PARCEL"]):
            return None          # le vendeur de Jadielle ne fait que remettre le colis
        clerks = {}
        for name in map_names():
            for obj in load_map(name).objects:
                sold = [i for i in HEALING_ITEMS if obj.mart and i in obj.mart]
                if sold and name not in clerks:
                    clerks[name] = (obj.name, sold[0])
        found = self._nearest(state, lambda sq: sq.map in clerks)
        if found is None:
            return None
        clerk, item = clerks[found.map]
        return Goal("buy", {"map": found.map, "clerk": clerk, "item": item,
                            "quantity": self.config.buy_potions})

    # ── Requêtes de carte ─────────────────────────────────────────────────────

    def _center_reachable(self, state: GameState) -> bool:
        return self._nearest(state, lambda sq: sq.map in nurse_maps()) is not None

    def training_map(self, state: GameState) -> str | None:
        """Carte avec de l'herbe la plus proche (en cases), Pokémon sauvages pas trop forts."""
        level = max_level(state)
        maps = {}
        for name in map_names():
            data = load_map(name)
            if data.wild and data.wild.grass_rate and max(lv for lv, _ in data.wild.grass) <= level + 2:
                squares = grass_squares(name)
                if squares:
                    maps[name] = set(squares)
        found = self._nearest(state, lambda sq: (sq.x, sq.y) in maps.get(sq.map, ()))
        return found.map if found else None

    @staticmethod
    def _nearest(state: GameState, predicate) -> Square | None:
        nav = Navigator.from_state(state)
        start = Square(_here(state), state.x, state.y)
        path = nav.search([start], predicate)
        return path.squares[-1] if path else None
