"""
Debug Visualizer — Pokémon Blue AI
Affiche en temps réel les entités détectées depuis la RAM (sprites + tiles).
Outil de diagnostic pour valider la lecture RAM avant d'entraîner les agents.

Raccourcis :
  p  — pause / reprise
  d  — dump JSON de la RAM courante dans debug/
  q  — quitter
"""

import json
import os

import cv2
import numpy as np

os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
from pyboy import PyBoy

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
ROM_PATH     = 'ROMs/PokemonBlue.gb'
MAPPING_PATH = 'mapping.json'
STATE        = 'states/07_route1_grass.state'
SCALE        = 3       # Facteur d'agrandissement fenêtre OpenCV
DEBUG_DIR    = 'debug'

SCREEN_W, SCREEN_H = 160, 144

# Couleurs BGR par type d'entité
CLASS_COLORS = {
    0: (255, 100,   0),   # Player — bleu
    1: (  0,   0, 255),   # NPC    — rouge
    2: (255, 255,   0),   # Door   — cyan
    3: (  0, 200,   0),   # Grass  — vert
    4: (  0, 215, 255),   # Ledge  — jaune
    5: (255,   0, 255),   # Item   — magenta
}
CLASS_NAMES = {0: 'Player', 1: 'NPC', 2: 'Door', 3: 'Grass', 4: 'Ledge', 5: 'Item'}

# ---------------------------------------------------------------------------
# MAPPING
# ---------------------------------------------------------------------------

def load_mapping(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"mapping.json introuvable : {path}")
    with open(path) as f:
        return json.load(f)


def build_tile_lookup(mapping, tileset_name):
    lookup = {}
    tiles = mapping.get('tiles', {})
    for category in ('doors', 'tall_grass', 'ledges'):
        cat = tiles.get(category, {})
        for hex_id, info in cat.get(tileset_name, {}).items():
            if not hex_id.startswith('_'):
                lookup[hex_id] = info
    return lookup

# ---------------------------------------------------------------------------
# OAM  (filtre sprites fantômes)
# ---------------------------------------------------------------------------

def get_oam_tiles(pyboy):
    """
    Retourne le set de tuiles 8×8 (tx, ty) rendues en avant-plan par le hardware.
    Sprite avec bit 7 attribut = 1 (derrière le fond) → ignoré.
    """
    occupied = set()
    for i in range(40):
        base     = 0xFE00 + i * 4
        oam_y    = pyboy.memory[base]
        oam_x    = pyboy.memory[base + 1]
        oam_attr = pyboy.memory[base + 3]
        if oam_attr & 0x80:
            continue
        if 16 <= oam_y < 160 and 8 <= oam_x < 168:
            occupied.add(((oam_x - 8) >> 3, (oam_y - 16) >> 3))
    return occupied

# ---------------------------------------------------------------------------
# SCAN SPRITES  (table WRAM 0xC100)
# ---------------------------------------------------------------------------

OVERWORLD_BLOCK_TILES = {
    0x00, 0x10, 0x1B, 0x20, 0x21, 0x23, 0x2C, 0x2D, 0x2E,
    0x30, 0x31, 0x33, 0x39, 0x3C, 0x3E, 0x54, 0x58, 0x5B,
}


def scan_sprites(dashboard, pyboy, mapping):
    """Dessine les sprites actifs lus depuis WRAM 0xC100. Retourne un dict de comptages."""
    counts  = {}
    sprites = mapping.get('sprites', {})
    oam_tiles = get_oam_tiles(pyboy)

    for slot in range(16):
        base   = 0xC100 + slot * 16
        pic_id = pyboy.memory[base + 0x00]
        mov_st = pyboy.memory[base + 0x01]

        if pic_id == 0:
            continue

        if slot == 0:
            sx, sy = 64, 64
        else:
            if mov_st == 0:
                continue
            sx = pyboy.memory[base + 0x06]
            sy = pyboy.memory[base + 0x04]
            if sx == 0 and sy == 0:
                continue
            if not (0 <= sx < SCREEN_W and 0 <= sy < SCREEN_H):
                continue
            tile_x, tile_y = sx >> 3, sy >> 3
            if not any(
                (tile_x + dtx, tile_y + dty) in oam_tiles
                for dtx in range(-1, 3) for dty in range(-1, 3)
            ):
                continue

        hex_id = f"0x{pic_id:02X}"
        info   = sprites.get(hex_id)
        if info:
            cls   = info['class']
            color = CLASS_COLORS.get(cls, (200, 200, 200))
            cv2.rectangle(dashboard, (sx, sy), (sx + 16, sy + 16), color, 1)
            cv2.putText(dashboard, info['name'], (sx, max(sy - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)
            counts[CLASS_NAMES.get(cls, f'cls{cls}')] = counts.get(CLASS_NAMES.get(cls, f'cls{cls}'), 0) + 1
        else:
            cv2.rectangle(dashboard, (sx, sy), (sx + 16, sy + 16), (0, 0, 200), 1)
            cv2.putText(dashboard, f"?{hex_id}", (sx, max(sy - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 200), 1)

    return counts


def scan_tiles(dashboard, pyboy, tile_lookup):
    """Dessine les tiles actifs lus depuis WRAM 0xC3A0. Retourne un dict de comptages."""
    counts = {}

    for row in range(18):
        for col in range(20):
            tid    = pyboy.memory[0xC3A0 + row * 20 + col]
            hex_id = f"0x{tid:02X}"
            info   = tile_lookup.get(hex_id)
            if not info:
                continue

            cls   = info['class']
            color = CLASS_COLORS.get(cls, (200, 200, 200))

            if cls == 4:  # Ledge — filtre mur de montagne
                if hex_id in {'0x36', '0x37'}:
                    if row > 0 and pyboy.memory[0xC3A0 + (row - 1) * 20 + col] in OVERWORLD_BLOCK_TILES:
                        continue
                elif hex_id == '0x27':
                    if col < 19 and pyboy.memory[0xC3A0 + row * 20 + (col + 1)] in OVERWORLD_BLOCK_TILES:
                        continue
                elif hex_id in {'0x0D', '0x1D'}:
                    if col > 0 and pyboy.memory[0xC3A0 + row * 20 + (col - 1)] in OVERWORLD_BLOCK_TILES:
                        continue

            if cls == 2:  # Door — box 16×16 décalée vers le haut
                px  = col * 8
                py_ = max(row * 8 - 8, 0)
                cv2.rectangle(dashboard,
                              (px, py_),
                              (min(px + 16, SCREEN_W), min(py_ + 16, SCREEN_H)),
                              color, 2)
            else:
                px, py_ = col * 8, row * 8
                cv2.rectangle(dashboard, (px, py_), (px + 8, py_ + 8), color, 1)

            name = CLASS_NAMES.get(cls, f'cls{cls}')
            counts[name] = counts.get(name, 0) + 1

    return counts

# ---------------------------------------------------------------------------
# PANNEAU LATÉRAL
# ---------------------------------------------------------------------------

def draw_panel(dashboard, pyboy, mapping, m_id, t_id, px, py, counts, frame):
    ox = 168
    lh = 14

    tileset_name = mapping.get('tilesets', {}).get(f"0x{t_id:02X}", f"?({t_id})")
    battle       = pyboy.memory[0xD057]
    grass_flag   = pyboy.memory[0xC207]
    direction    = pyboy.memory[0xD35D]
    dir_name     = {0x00: 'DOWN', 0x04: 'UP', 0x08: 'LEFT', 0x0C: 'RIGHT'}.get(direction, f"0x{direction:02X}")
    badges       = pyboy.memory[0xD356]
    tile_under   = pyboy.memory[0xC3A0 + 9 * 20 + 9]

    lines = [
        (f"frame  {frame}", (200, 200, 200)),
        ("", None),
        (f"map    0x{m_id:02X}", (200, 200, 200)),
        (f"tset   0x{t_id:02X} {tileset_name}", (200, 200, 200)),
        (f"pos    x={px} y={py}", (200, 200, 200)),
        (f"dir    {dir_name}", (200, 200, 200)),
        ("", None),
        (f"battle 0x{battle:02X}", (0, 100, 255) if battle else (200, 200, 200)),
        (f"grass  {'IN' if grass_flag == 0x80 else 'out'}", (0, 200, 0) if grass_flag == 0x80 else (200, 200, 200)),
        (f"t@72,72 0x{tile_under:02X}", (180, 180, 255)),
        (f"badges 0b{badges:08b}", (255, 200, 0) if badges else (200, 200, 200)),
        ("", None),
        ("--- détections ---", (120, 120, 120)),
    ]

    for name, n in counts.items():
        cls_id = next((k for k, v in CLASS_NAMES.items() if v == name), -1)
        color  = CLASS_COLORS.get(cls_id, (200, 200, 200))
        lines.append((f"{name:<8} {n}", color))

    lines += [
        ("", None),
        ("--- keys ---", (120, 120, 120)),
        ("p  pause", (160, 160, 160)),
        ("d  dump RAM", (160, 160, 160)),
        ("q  quit", (160, 160, 160)),
    ]

    for i, (text, color) in enumerate(lines):
        if text and color:
            cv2.putText(dashboard, text, (ox, 14 + i * lh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    mapping = load_mapping(MAPPING_PATH)
    pyboy   = PyBoy(ROM_PATH, window="SDL2", sound=False)
    pyboy.set_emulation_speed(1)

    if STATE and os.path.exists(STATE):
        with open(STATE, 'rb') as f:
            pyboy.load_state(f)
        for _ in range(5):
            pyboy.tick()
        print(f"State chargé : {STATE}")
    else:
        print(f"Aucun state trouvé ({STATE}), démarrage à froid.")

    win = 'RAM Debug — Pokémon Blue AI'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, (SCREEN_W + 160) * SCALE, SCREEN_H * SCALE)

    frame_count = 0
    paused      = False
    print("Raccourcis : p=pause  d=dump  q=quit")

    try:
        while True:
            if not paused:
                if not pyboy.tick():
                    break

            screen = pyboy.screen.image
            if getattr(screen, 'mode', None) == 'RGBA':
                screen = screen.convert('RGB')

            raw       = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
            dashboard = cv2.copyMakeBorder(
                raw.copy(), 0, 0, 0, 160, cv2.BORDER_CONSTANT, value=(30, 30, 30)
            )

            m_id = pyboy.memory[0xD35E]
            t_id = pyboy.memory[0xD367]
            px   = pyboy.memory[0xD362]
            py   = pyboy.memory[0xD361]

            tileset_name = mapping.get('tilesets', {}).get(f"0x{t_id:02X}", 'OVERWORLD')
            tile_lookup  = build_tile_lookup(mapping, tileset_name)

            counts  = scan_sprites(dashboard, pyboy, mapping)
            tile_counts = scan_tiles(dashboard, pyboy, tile_lookup)
            for k, v in tile_counts.items():
                counts[k] = counts.get(k, 0) + v

            draw_panel(dashboard, pyboy, mapping, m_id, t_id, px, py, counts, frame_count)

            out = cv2.resize(
                dashboard,
                (dashboard.shape[1] * SCALE, dashboard.shape[0] * SCALE),
                interpolation=cv2.INTER_NEAREST
            )
            cv2.imshow(win, out)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord('p'):
                paused = not paused
                print("Pause" if paused else "Reprise")
            elif key == ord('d'):
                os.makedirs(DEBUG_DIR, exist_ok=True)
                dump = {
                    'frame':       frame_count,
                    'map_id':      m_id,
                    'tileset':     t_id,
                    'tileset_name': tileset_name,
                    'pos':         {'x': px, 'y': py},
                    'battle':      pyboy.memory[0xD057],
                    'badges':      pyboy.memory[0xD356],
                    'grass':       hex(pyboy.memory[0xC207]),
                    'wram_sprites': {
                        slot: {
                            'pic_id':   hex(pyboy.memory[0xC100 + slot * 16]),
                            'mov_stat': pyboy.memory[0xC100 + slot * 16 + 0x01],
                            'sy':       pyboy.memory[0xC100 + slot * 16 + 0x04],
                            'sx':       pyboy.memory[0xC100 + slot * 16 + 0x06],
                        }
                        for slot in range(16)
                        if pyboy.memory[0xC100 + slot * 16] != 0
                    },
                }
                path = os.path.join(DEBUG_DIR, f"dump_{frame_count:05d}.json")
                with open(path, 'w') as f:
                    json.dump(dump, f, indent=2)
                print(f"Dump RAM : {path}")

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        pyboy.stop()
        cv2.destroyAllWindows()
        print("Terminé.")


if __name__ == '__main__':
    main()
