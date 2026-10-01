"""Lecture de l'écran à partir du tilemap (wTileMap) : texte, boîtes et curseurs.

Le tilemap contient les tuiles de l'écran. Dans l'overworld, ce sont uniquement des
tuiles de décor (jeu de tuiles chargé à partir de la tuile 0) ; dès qu'une boîte de
texte, un menu ou un écran d'interface est affiché, on y trouve les tuiles de police
et de cadre définies par charmap.asm.
"""

from __future__ import annotations

from pokeblue.knowledge.gen1_data import CHARMAP, TILE_CHARS, decode_tiles
from pokeblue.state import ram_symbols as sym

WIDTH, HEIGHT = sym.SCREEN_WIDTH, sym.SCREEN_HEIGHT

CURSOR = CHARMAP["▶"]             # curseur de menu actif
CURSOR_INACTIVE = CHARMAP["▷"]    # curseur d'un menu parent resté affiché
TEXT_ARROW = CHARMAP["▼"]         # invite « appuyer sur A » (clignote)
BOX_TOP_LEFT, BOX_TOP_RIGHT = CHARMAP["┌"], CHARMAP["┐"]
BOX_BOTTOM_LEFT, BOX_BOTTOM_RIGHT = CHARMAP["└"], CHARMAP["┘"]
BOX_TILES = frozenset(CHARMAP[c] for c in "┌─┐│└┘")

# Tuiles d'interface : cadres et police. Le décor de l'overworld n'utilise que les
# tuiles situées en dessous (jeu de tuiles de la carte).
UI_TILES_START = min(BOX_TILES | TILE_CHARS.keys())

# Boîte de texte standard : toute la largeur, sur les 6 dernières lignes.
TEXT_BOX_TOP = HEIGHT - 6


class Screen:
    """Vue en lecture seule sur un tilemap de SCREEN_WIDTH × SCREEN_HEIGHT tuiles."""

    __slots__ = ("tiles",)

    def __init__(self, tiles: bytes) -> None:
        if len(tiles) != WIDTH * HEIGHT:
            raise ValueError(f"tilemap de {len(tiles)} tuiles")
        self.tiles = tiles

    def tile(self, x: int, y: int) -> int:
        return self.tiles[y * WIDTH + x]

    def row(self, y: int) -> bytes:
        return self.tiles[y * WIDTH:(y + 1) * WIDTH]

    def text_rows(self, unknown: str = "·") -> list[str]:
        return [decode_tiles(self.row(y), unknown) for y in range(HEIGHT)]

    def contains_text(self, text: str) -> bool:
        return any(text in line for line in self.text_rows())

    @property
    def ui_tile_count(self) -> int:
        return sum(t >= UI_TILES_START for t in self.tiles)

    @property
    def has_cursor(self) -> bool:
        return CURSOR in self.tiles

    def cursor_positions(self) -> list[tuple[int, int]]:
        return [(i % WIDTH, i // WIDTH) for i, t in enumerate(self.tiles) if t == CURSOR]

    def has_box(self, left: int, top: int, right: int, bottom: int) -> bool:
        """Vrai si les quatre coins d'un cadre sont aux positions données."""
        return (
            self.tile(left, top) == BOX_TOP_LEFT
            and self.tile(right, top) == BOX_TOP_RIGHT
            and self.tile(left, bottom) == BOX_BOTTOM_LEFT
            and self.tile(right, bottom) == BOX_BOTTOM_RIGHT
        )

    @property
    def has_text_box(self) -> bool:
        """Boîte de texte standard (bas de l'écran, toute la largeur)."""
        return self.has_box(0, TEXT_BOX_TOP, WIDTH - 1, HEIGHT - 1)
