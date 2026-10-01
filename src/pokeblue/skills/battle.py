"""BattleSkill (baseline scriptée) : mène un combat de bout en bout.

- Messages et animations : A (jamais B, qui annulerait une évolution).
- Choix de l'attaque : multiplicateur de type Gen 1 × puissance × STAB × précision ×
  rapport Attaque/Défense (ou Spécial/Spécial selon le type), PP > 0.
- Fuite : combats sauvages quand l'objectif le demande (`flee_wild`) ou quand le
  Pokémon actif passe sous `flee_below` de ses PV sans objet de soin.
- Soin : Potion (ou mieux) sur le Pokémon actif sous `heal_below` de ses PV.
- K.O. : envoi du premier Pokémon valide ; « Use next POKéMON? » → OUI.
- Nouvelle attaque : apprise seulement si elle fait mieux que la plus faible connue,
  qui est alors oubliée ; sinon abandonnée.
"""

from __future__ import annotations

from pokeblue.knowledge.gen1_data import ITEM_IDS, MOVE_IDS, MOVES, is_special_type, type_multiplier
from pokeblue.skills.base import BaseSkill, Button, Goal
from pokeblue.skills.menu_reader import Menu, read_menu, screen_text
from pokeblue.skills.menus import bag_button
from pokeblue.state.game_state import BattleMon, GameState
from pokeblue.state.mode_detector import Mode, detect_mode
from pokeblue.state.screen import Screen

HEALING_ITEMS = ("FULL_RESTORE", "MAX_POTION", "HYPER_POTION", "SUPER_POTION", "POTION")
FIXED_DAMAGE = {"SONICBOOM": 20, "DRAGON_RAGE": 40}
LEVEL_DAMAGE = {"SEISMIC_TOSS", "NIGHT_SHADE"}
STATUS_MOVE_SCORE = 0.1   # une attaque sans dégâts n'est choisie que faute de mieux


def move_score(move_id: int, user: BattleMon, target: BattleMon) -> float:
    """Dégâts relatifs attendus d'une attaque (pour classer les attaques, pas les prédire)."""
    move = MOVES[move_id]
    if move.power == 0:
        return STATUS_MOVE_SCORE
    mult = type_multiplier(move.type, target.types)
    if mult == 0:
        return 0.0
    accuracy = move.accuracy / 100
    if move.name in FIXED_DAMAGE:
        return FIXED_DAMAGE[move.name] * accuracy
    if move.name in LEVEL_DAMAGE:
        return user.level * accuracy
    if is_special_type(move.type):
        ratio = user.special / max(target.special, 1)
    else:
        ratio = user.attack / max(target.defense, 1)
    stab = 1.5 if move.type in user.types else 1.0
    return move.power * mult * stab * accuracy * ratio


def best_move_index(user: BattleMon, target: BattleMon, disabled_slot: int = -1) -> int:
    scores = [
        move_score(m, user, target) if m and pp > 0 and i != disabled_slot else -1.0
        for i, (m, pp) in enumerate(zip(user.moves, user.pp, strict=True))
    ]
    return max(range(len(scores)), key=lambda i: scores[i])


class BattleSkill(BaseSkill):
    name = "battle"
    budget_steps = 3000

    def start(self, goal: Goal, state: GameState) -> None:
        super().start(goal, state)
        self.pending_heal = False        # menu principal → sac pour se soigner
        self.heal_on_active = False      # objet choisi : la cible est le Pokémon actif

    # ── Décision ──────────────────────────────────────────────────────────────

    def _act(self, state: GameState) -> Button | None:
        mode = detect_mode(state)
        if mode is Mode.TRANSITION:
            return None
        if mode is Mode.BATTLE_ANIM or mode is Mode.DIALOG:
            return "a"
        screen = Screen(state.tilemap)
        menu = read_menu(screen)
        if menu is None:
            return "a"
        if mode is Mode.BATTLE_MOVE_MENU and state.battle:
            battle = state.battle
            return menu.press_towards(best_move_index(battle.player, battle.enemy,
                                                      battle.player_disabled_slot))
        return self._menu(state, screen, menu)

    def _menu(self, state: GameState, screen: Screen, menu: Menu) -> Button:
        text = screen_text(screen).upper()
        if menu.index_of("YES") >= 0 and menu.index_of("NO") >= 0:
            return self._yes_no(state, text, menu)
        if screen.contains_text("FIGHT"):
            return self._main_menu(state, screen)
        if menu.index_of("SWITCH") >= 0:            # sous-menu d'un Pokémon de l'équipe
            return menu.press_towards(menu.index_of("SWITCH"))
        if "FORGOTTEN" in text and state.battle:    # quelle attaque oublier ?
            return menu.press_towards(self._move_to_forget(state.battle.player))
        if menu.cursor_x == 0 and state.party:       # liste de l'équipe
            return self._choose_party_member(state, menu)
        if self.pending_heal:                         # sac ouvert pour se soigner
            item = next((ITEM_IDS[i] for i in HEALING_ITEMS if state.item_count(ITEM_IDS[i])), None)
            if item is not None:
                button = bag_button(menu, state, item)    # la liste défile par 4
                if button == "a":
                    self.pending_heal, self.heal_on_active = False, True
                return button
        return "b"

    def _yes_no(self, state: GameState, text: str, menu: Menu) -> Button:
        no = menu.index_of("NO")
        if "NICKNAME" in text:
            return menu.press_towards(no)
        if "MAKE ROOM" in text or "DELETE AN OLDER MOVE" in text:
            learn = self._new_move(text)
            worth = learn is not None and state.battle and self._learn_is_better(state.battle.player, learn)
            return menu.press_towards(menu.index_of("YES") if worth else no)
        return menu.press_towards(menu.index_of("YES"))   # Use next POKéMON?, abandonner…

    def _main_menu(self, state: GameState, screen: Screen) -> Button:
        positions = {}
        for word in ("FIGHT", "ITEM", "RUN"):
            for y, row in enumerate(screen.text_rows(" ")):
                x = row.find(word)
                if x >= 0:
                    positions[word] = (x - 1, y)
        if len(positions) < 3 or not screen.cursor_positions():
            return "a"
        positions["PKMN"] = (positions["RUN"][0], positions["FIGHT"][1])
        target = "FIGHT"
        battle = state.battle
        if battle and self._wants_heal(state):
            target = "ITEM"
            self.pending_heal = True
        elif battle and battle.is_wild and self._wants_flee(state):
            target = "RUN"
        cx, cy = screen.cursor_positions()[0]
        tx, ty = positions[target]
        if (cx, cy) == (tx, ty):
            return "a"
        if cy != ty:
            return "down" if ty > cy else "up"
        return "right" if tx > cx else "left"

    def _wants_flee(self, state: GameState) -> bool:
        if self.goal is None:
            return False
        mon = state.battle.player
        low = mon.max_hp and mon.hp / mon.max_hp < self.goal.get("flee_below", 0.0)
        return bool(self.goal.get("flee_wild") or low)

    def _wants_heal(self, state: GameState) -> bool:
        threshold = self.goal.get("heal_below", 0.3) if self.goal else 0.3
        mon = state.battle.player
        if not mon.max_hp or mon.hp / mon.max_hp >= threshold:
            return False
        return any(state.item_count(ITEM_IDS[i]) for i in HEALING_ITEMS)

    def _choose_party_member(self, state: GameState, menu: Menu) -> Button:
        current = menu.cursor_y // 2
        active = state.battle.player_party_index if state.battle else -1
        candidates = [i for i, mon in enumerate(state.party) if mon.hp > 0]
        if self.heal_on_active or not candidates:
            target = active if active >= 0 else current
        else:
            target = next((i for i in candidates if i != active), candidates[0])
        if current == target:
            self.heal_on_active = False
            return "a"
        return "down" if target > current else "up"

    # ── Apprentissage d'attaques ──────────────────────────────────────────────

    @staticmethod
    def _new_move(text: str) -> int | None:
        compact = text.replace(" ", "_")
        found = [mid for name, mid in MOVE_IDS.items() if name in compact or name.replace("_", "") in text.replace(" ", "")]
        return max(found, key=lambda m: len(MOVES[m].name)) if found else None

    @staticmethod
    def _weakness(move_id: int, user: BattleMon) -> float:
        move = MOVES[move_id]
        if move.power == 0:
            return 0.0
        stab = 1.5 if move.type in user.types else 1.0
        return move.power * stab * move.accuracy / 100

    def _learn_is_better(self, user: BattleMon, new: int) -> bool:
        known = [m for m in user.moves if m]
        worst = min(known, key=lambda m: self._weakness(m, user))
        return self._weakness(new, user) > self._weakness(worst, user)

    def _move_to_forget(self, user: BattleMon) -> int:
        return min(range(len(user.moves)), key=lambda i: self._weakness(user.moves[i], user)
                   if user.moves[i] else -1)

    def _succeeded(self, state: GameState) -> bool:
        return state.battle is None and detect_mode(state) is Mode.OVERWORLD
