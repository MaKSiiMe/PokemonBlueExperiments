"""DialogSkill (baseline scriptée) : fait défiler les textes et répond aux questions.

Politique hors combat :
- texte : A ;
- question OUI/NON : NON pour un surnom, OUI sinon (prendre un objet, choisir un
  starter, accepter un échange de script…), ou la réponse imposée par l'objectif ;
- infirmière (SOIGNER/ANNULER) : SOIGNER ;
- autre menu inattendu (menu Start resté ouvert, liste…) : B pour le refermer.

Le skill réussit dès que le jeu est revenu dans l'overworld.
"""

from __future__ import annotations

from pokeblue.skills.base import BaseSkill, Button
from pokeblue.skills.menu_reader import read_menu, screen_text
from pokeblue.state.game_state import GameState
from pokeblue.state.mode_detector import Mode, detect_mode
from pokeblue.state.screen import Screen


def answer_yes_no(text: str, prefer: str | None = None) -> str:
    """Réponse à une question OUI/NON affichée (« YES » ou « NO »)."""
    if prefer:
        return prefer
    return "NO" if "NICKNAME" in text.upper() else "YES"


def dialog_button(state: GameState, prefer: str | None = None) -> Button | None:
    """Bouton à presser face au texte ou au menu affiché (None : rien à faire)."""
    mode = detect_mode(state)
    if mode is Mode.TRANSITION or mode is Mode.OVERWORLD:
        return None
    if mode is Mode.DIALOG:
        return "a"
    screen = Screen(state.tilemap)
    menu = read_menu(screen)
    if menu is None:
        return "a"
    yes, no = menu.index_of("YES"), menu.index_of("NO")
    if yes >= 0 and no >= 0:
        answer = answer_yes_no(screen_text(screen), prefer)
        return menu.press_towards(yes if answer == "YES" else no)
    heal = menu.index_of("HEAL")
    if heal >= 0 and menu.index_of("CANCEL") >= 0:
        return menu.press_towards(heal)
    return "b"


class DialogSkill(BaseSkill):
    name = "dialog"
    budget_steps = 400

    def _act(self, state: GameState) -> Button | None:
        return dialog_button(state, self.goal.get("answer") if self.goal else None)

    def _succeeded(self, state: GameState) -> bool:
        return self.steps > 0 and detect_mode(state) is Mode.OVERWORLD
