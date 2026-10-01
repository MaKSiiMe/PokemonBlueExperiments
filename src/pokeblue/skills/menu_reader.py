"""Lecture générique des menus à l'écran : curseur ▶, options, déplacement.

Un menu Gen 1 est un cadre (┌─┐ │ └─┘) dont les options sont alignées sur la colonne
qui suit le curseur ▶. On lit les options dans le tilemap (texte décodé par la
charmap pokered) et on calcule les pressions nécessaires pour atteindre une option.
"""

from __future__ import annotations

from dataclasses import dataclass

from pokeblue.knowledge.gen1_data import TILE_CHARS
from pokeblue.state.screen import CURSOR, HEIGHT, WIDTH, Screen

BOX_HORIZONTAL = {c for c in "─┌┐└┘"}


@dataclass(frozen=True)
class MenuOption:
    row: int
    text: str


@dataclass(frozen=True)
class Menu:
    cursor_x: int
    cursor_y: int
    options: tuple[MenuOption, ...]

    @property
    def selected(self) -> int:
        return next((i for i, o in enumerate(self.options) if o.row == self.cursor_y), -1)

    def index_of(self, *needles: str) -> int:
        """Indice de la première option qui contient l'un des textes (casse ignorée)."""
        for i, option in enumerate(self.options):
            if any(n.upper() in option.text.upper() for n in needles):
                return i
        return -1

    def press_towards(self, index: int) -> str:
        """Bouton pour rapprocher le curseur de l'option `index` (« a » si on y est)."""
        current = self.selected
        if current == index:
            return "a"
        return "down" if index > current else "up"


def _char(screen: Screen, x: int, y: int) -> str:
    return TILE_CHARS.get(screen.tile(x, y), "·")


def _row_text(screen: Screen, x0: int, y: int) -> str:
    chars = []
    for x in range(x0, WIDTH):
        c = _char(screen, x, y)
        if c == "│":
            break
        chars.append(c)
    return "".join(chars).rstrip(" ·")


def read_menu(screen: Screen, cursor: tuple[int, int] | None = None) -> Menu | None:
    """Menu vertical autour du curseur ▶ (le premier trouvé si non précisé)."""
    positions = screen.cursor_positions()
    if cursor is None:
        if not positions:
            return None
        cursor = positions[0]
    cx, cy = cursor
    top = cy
    while top > 0 and _char(screen, cx, top - 1) not in BOX_HORIZONTAL:
        top -= 1
    bottom = cy
    while bottom < HEIGHT - 1 and _char(screen, cx, bottom + 1) not in BOX_HORIZONTAL:
        bottom += 1
    options = []
    for y in range(top, bottom + 1):
        if screen.tile(cx, y) == CURSOR or _char(screen, cx, y) in (" ", "▷"):
            text = _row_text(screen, cx + 1, y).strip()
            # Les lignes de quantité (× 2), de prix (¥200) ou de PV d'une liste ne sont
            # pas des options : une option commence par une lettre, un chiffre ou « - ».
            if text and (text[0].isalnum() or text[0] == "-"):
                options.append(MenuOption(y, text))
    return Menu(cx, cy, tuple(options))


def screen_text(screen: Screen) -> str:
    """Tout le texte affiché, lignes jointes (pour reconnaître une question)."""
    return " ".join(row.strip(" ·") for row in screen.text_rows(" ") if row.strip(" ·"))
