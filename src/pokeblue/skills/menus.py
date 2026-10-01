"""MenuSkills (macros scriptées) : apprendre une CS/CT, utiliser une capacité de terrain,
acheter en boutique.

Les menus sont reconnus à l'écran (pokeblue.skills.menu_reader) : menu Start, sac,
USE/TOSS, liste de l'équipe, sous-menu d'un Pokémon, question OUI/NON, choix de
l'attaque à oublier. Chaque macro est une petite machine à états pilotée par ce
qu'affiche l'écran, jamais par une séquence de boutons aveugle.
"""

from __future__ import annotations

from pokeblue.knowledge.gen1_data import ITEM_IDS, ITEMS, MOVE_IDS, MOVES
from pokeblue.skills.base import BaseSkill, Button, Goal, SkillStatus
from pokeblue.skills.dialog import dialog_button
from pokeblue.skills.menu_reader import Menu, read_menu, screen_text
from pokeblue.skills.navigation import Navigating, NavigationSkill
from pokeblue.state.game_state import GameState, PartyMon
from pokeblue.state.mode_detector import Mode, detect_mode
from pokeblue.state.screen import Screen

FIRST_HM, FIRST_TM = ITEM_IDS["HM_CUT"], ITEM_IDS["TM_MEGA_PUNCH"]   # HM01, TM01


def item_label(item_id: int) -> str:
    """Nom affiché dans le sac (HM01, TM34, DOME FOSSIL…)."""
    if FIRST_HM <= item_id < FIRST_TM:
        return f"HM{item_id - FIRST_HM + 1:02d}"
    if item_id >= FIRST_TM:
        return f"TM{item_id - FIRST_TM + 1:02d}"
    return ITEMS[item_id].replace("_", " ")


def bag_button(menu: Menu, state: GameState, item_id: int) -> Button:
    """Rapproche le curseur du sac de l'objet voulu (la liste défile par 4)."""
    labels = [item_label(item) for item, _ in state.bag] + ["CANCEL"]
    target = next(i for i, (item, _) in enumerate(state.bag) if item == item_id)
    if not menu.options:
        return "down"
    first = next((i for i, label in enumerate(labels) if label == menu.options[0].text), None)
    if first is None:
        return "down"
    current = first + max(menu.selected, 0)
    if current == target:
        return "a"
    return "down" if target > current else "up"


def knows(mon: PartyMon, move: str) -> bool:
    return MOVE_IDS[move] in mon.moves


def weakest_move_slot(mon: PartyMon) -> int:
    """Attaque à oublier : la moins utile en dégâts (les attaques de statut d'abord)."""
    def value(slot: int) -> float:
        move = MOVES[mon.moves[slot]] if mon.moves[slot] else None
        if move is None:
            return -1.0
        stab = 1.5 if move.type in mon.types else 1.0
        return move.power * stab * move.accuracy / 100
    return min(range(len(mon.moves)), key=value)


def _start_menu(menu: Menu) -> bool:
    return menu.index_of("ITEM") >= 0 and menu.index_of("EXIT") >= 0


def _party_list(menu: Menu, state: GameState) -> bool:
    return menu.cursor_x == 0 and bool(state.party)


def _party_button(menu: Menu, index: int) -> Button:
    current = menu.cursor_y // 2         # un Pokémon toutes les deux lignes
    if current == index:
        return "a"
    return "down" if index > current else "up"


class TeachMoveSkill(BaseSkill):
    """Apprendre la CS/CT `item` (attaque `move`) au Pokémon `party_index` (défaut 0)."""

    name = "teach"
    budget_steps = 300
    handles_menus = True

    def _act(self, state: GameState) -> Button | None:
        index = self.goal.get("party_index", 0)
        mon = state.party[index]
        mode = detect_mode(state)
        if mode is Mode.TRANSITION:
            return None
        if knows(mon, self.goal["move"]):
            self.done = mode is Mode.OVERWORLD
            return None if self.done else "b"
        if mode is Mode.OVERWORLD:
            if not state.item_count(ITEM_IDS[self.goal["item"]]):
                self.fail(f"{self.goal['item']} absent du sac")
                return None
            return "start"
        if mode is Mode.DIALOG:
            return "a"
        screen = Screen(state.tilemap)
        menu = read_menu(screen)
        if menu is None:
            return "a"
        text = screen_text(screen).upper()
        if menu.index_of("YES") >= 0 and menu.index_of("NO") >= 0:
            return dialog_button(state)                         # apprendre / effacer : OUI
        if "FORGOTTEN" in text:
            return menu.press_towards(weakest_move_slot(mon))
        if _start_menu(menu):
            return menu.press_towards(menu.index_of("ITEM"))
        if menu.index_of("USE") >= 0 and menu.index_of("TOSS") >= 0:
            return menu.press_towards(menu.index_of("USE"))
        if _party_list(menu, state):
            return _party_button(menu, index)
        return bag_button(menu, state, ITEM_IDS[self.goal["item"]])


class FieldMoveSkill(BaseSkill):
    """Utiliser une capacité de terrain (CUT, SURF, STRENGTH…) depuis le menu Équipe.

    Le joueur doit déjà faire face à l'obstacle (NavigationSkill s'en charge)."""

    name = "field_move"
    budget_steps = 120
    handles_menus = True

    def start(self, goal: Goal, state: GameState) -> None:
        super().start(goal, state)
        self.used = False

    def _act(self, state: GameState) -> Button | None:
        mode = detect_mode(state)
        move = self.goal["move"]
        if mode is Mode.TRANSITION or (mode is Mode.OVERWORLD and state.scripted):
            return None
        if mode is Mode.OVERWORLD:
            if self.used:
                self.done = True
                return None
            if not any(knows(mon, move) for mon in state.party):
                self.fail(f"aucun Pokémon ne connaît {move}")
                return None
            return "start"
        if mode is Mode.DIALOG:
            return "a"
        screen = Screen(state.tilemap)
        menu = read_menu(screen)
        if menu is None:
            return "a"
        if self.used:
            return "b"
        if _start_menu(menu):
            return menu.press_towards(menu.index_of("MON"))      # POKéMON
        sub = menu.index_of(move)
        if sub >= 0 and menu.cursor_x > 0:
            button = menu.press_towards(sub)
            self.used = button == "a"
            return button
        if _party_list(menu, state):
            index = next(i for i, mon in enumerate(state.party) if knows(mon, move))
            return _party_button(menu, index)
        return "b"


def _price_below(screen: Screen, row: int) -> int | None:
    """Prix affiché sous une ligne de la liste d'une boutique (« ¥300 »)."""
    if row + 1 >= len(screen.text_rows()):
        return None
    text = screen.text_rows(" ")[row + 1]
    digits = "".join(c for c in text.split("¥")[-1] if c.isdigit()) if "¥" in text else ""
    return int(digits) if digits else None


class BuySkill(Navigating, BaseSkill):
    """Acheter `quantity` exemplaires de `item` auprès du vendeur `clerk` de `map`.

    Achat à l'unité (quantité 1 validée par A) : plus lent qu'une saisie de quantité,
    mais sans état caché. S'arrête quand l'argent ne suffit plus."""

    name = "buy"
    budget_steps = 1500
    handles_menus = True

    def start(self, goal: Goal, state: GameState) -> None:
        super().start(goal, state)
        self.nav = NavigationSkill()
        self.nav_started = False
        self.item = ITEM_IDS[goal["item"]]
        self.target = state.item_count(self.item) + goal.get("quantity", 1)
        self.finished = False                  # assez acheté (ou plus d'argent) : sortir
        self.talked = False

    def _act(self, state: GameState) -> Button | None:
        mode = detect_mode(state)
        if mode is Mode.TRANSITION:
            return None
        if state.item_count(self.item) >= self.target:
            self.finished = True
        if mode is Mode.OVERWORLD:
            if self.talked and not state.scripted:
                self.done = True
                return None
            if not self.nav_started:
                self.nav.start(Goal("navigate", {"map": self.goal["map"], "object": self.goal["clerk"],
                                                 "interact": True}), state)
                self.nav_started = True
            button = self.nav.act(state)
            self.talked = self.nav.interacted
            if self.nav.status(state) in (SkillStatus.FAILURE, SkillStatus.TIMEOUT):
                self.fail(f"navigation : {self.nav.failure or 'timeout'}")
            return button
        if mode is Mode.DIALOG:
            return "a"
        screen = Screen(state.tilemap)
        menu = read_menu(screen)
        if menu is None:
            return "a"
        if menu.index_of("BUY") >= 0 and menu.index_of("QUIT") >= 0:
            return menu.press_towards(menu.index_of("QUIT" if self.finished else "BUY"))
        if menu.index_of("YES") >= 0 and menu.index_of("NO") >= 0:
            return menu.press_towards(menu.index_of("YES"))
        index = menu.index_of(item_label(self.item))
        if self.finished or index < 0:
            return "b"
        price = _price_below(screen, menu.options[index].row)
        if price is not None and state.money < price:
            self.finished = True
            return "b"
        return menu.press_towards(index)
