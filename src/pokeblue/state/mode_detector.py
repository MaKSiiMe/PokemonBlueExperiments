"""Détection du mode de jeu (overworld, dialogue, menus, combat, transition).

Il n'existe pas d'octet « dialogue actif » en Gen 1 : le mode est déduit de l'écran
(tilemap), des palettes et de wIsInBattle. Règles établies sur des traces frame par
frame et validées sur le jeu de savestates étiquetés de `configs/states/modes.yaml`
(tests/test_mode_detector.py) :

  TRANSITION  LCD éteint, ou palette de fond différente de la palette stable de la
              carte (fondus de porte, flash de début de combat), ou combat sans
              boîte de texte (spirale d'entrée en combat).
  Combat      curseur ▶ et cadre « TYPE/ » (ou « disabled! ») → BATTLE_MOVE_MENU ; curseur ▶ →
              BATTLE_MENU (menu principal, sac, équipe, oui/non) ; sinon BATTLE_ANIM
              (animations et messages de combat).
  Hors combat curseur ▶ → MENU ; tuiles d'interface (cadre, police) → DIALOG ;
              sinon OVERWORLD.
"""

from __future__ import annotations

from enum import Enum

from pokeblue.knowledge.gen1_data import FADE_PALETTES
from pokeblue.state.game_state import GameState
from pokeblue.state.screen import Screen

# LoadGBPal (home/fade.asm) lit la palette FadePal4 décalée de wMapPalOffset octets ;
# chaque FadePal fait 3 octets (rBGP, rOBP0, rOBP1).
_NORMAL_FADE_INDEX = 3   # FadePal4
_FADE_ENTRY_SIZE = 3

# Cadre d'information du menu des attaques : type de l'attaque sous le curseur, ou
# « disabled! » si elle est sous Entrave (DisabledText, engine/battle/core.asm).
MOVE_MENU_MARKERS = ("TYPE/", "disabled!")


class Mode(Enum):
    OVERWORLD = "overworld"
    DIALOG = "dialog"
    MENU = "menu"
    BATTLE_MENU = "battle_menu"
    BATTLE_MOVE_MENU = "battle_move_menu"
    BATTLE_ANIM = "battle_anim"
    TRANSITION = "transition"

    @property
    def in_battle(self) -> bool:
        return self.name.startswith("BATTLE")

    @property
    def accepts_menu_input(self) -> bool:
        return self in (Mode.MENU, Mode.BATTLE_MENU, Mode.BATTLE_MOVE_MENU)


def steady_bgp(map_pal_offset: int) -> int:
    """Palette de fond stable d'une carte selon wMapPalOffset (0 = normale)."""
    index = _NORMAL_FADE_INDEX - map_pal_offset // _FADE_ENTRY_SIZE
    return FADE_PALETTES[max(0, min(index, len(FADE_PALETTES) - 1))][0]


def detect_mode(state: GameState) -> Mode:
    screen = Screen(state.tilemap)
    if not state.lcd_on or state.bgp != steady_bgp(state.map_pal_offset):
        return Mode.TRANSITION

    if state.battle is not None:
        if screen.has_cursor:
            if any(screen.contains_text(marker) for marker in MOVE_MENU_MARKERS):
                return Mode.BATTLE_MOVE_MENU
            return Mode.BATTLE_MENU
        if screen.has_text_box:
            return Mode.BATTLE_ANIM
        return Mode.TRANSITION

    if screen.has_cursor:
        return Mode.MENU
    if screen.ui_tile_count:
        return Mode.DIALOG
    return Mode.OVERWORLD
