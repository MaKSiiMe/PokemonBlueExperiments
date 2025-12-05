import sys
import os
import glob
from pyboy import PyBoy
from PIL import ImageDraw

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
)

ROM_PATH = "PokemonBlue.gb"
OUTPUT_DIR = os.path.join("data", "debug_states")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ADDR_X_POS, ADDR_Y_POS, ADDR_MAP_N = 0xD362, 0xD361, 0xD35E
SCREEN_W, SCREEN_H = 160, 144

KNOWN_DOORS = {
    0:  [(5, 5), (13, 5), (12, 11)],
    1:  [(23, 25), (29, 19), (32, 7), (21, 15)],
    2:  [(13, 25), (23, 17), (16, 17), (7, 9)],
    33: [(8, 5)],
}


def draw_debug_info(pyboy, img, filename):
    draw = ImageDraw.Draw(img)
    map_id = pyboy.memory[ADDR_MAP_N]
    p_x, p_y = pyboy.memory[ADDR_X_POS], pyboy.memory[ADDR_Y_POS]

    if map_id in KNOWN_DOORS:
        for door_x, door_y in KNOWN_DOORS[map_id]:
            dx, dy = (door_x - p_x) * 16, (door_y - p_y) * 16
            screen_x, screen_y = 80 + dx, 72 + dy

            if -16 < screen_x < SCREEN_W and -16 < screen_y < SCREEN_H:
                draw.rectangle(
                    [screen_x, screen_y, screen_x + 16, screen_y + 16],
                    outline="yellow", width=2
                )
                draw.text((screen_x, screen_y - 10), "DOOR", fill="yellow")

    draw.text((5, 5), f"File: {filename}", fill="white")
    draw.text((5, 15), f"Map: {map_id} | Pos: ({p_x},{p_y})", fill="white")
    return img


def main():
    state_files = glob.glob("*.state")
    if not state_files:
        print("❌ Aucun fichier .state trouvé !")
        return

    print(f"🕵️  Audit de {len(state_files)} fichiers (Mode Stable)...")

    pyboy = PyBoy(ROM_PATH, window="null", sound=False)

    pyboy.set_emulation_speed(1)

    for state_file in state_files:
        try:
            with open(state_file, "rb") as f:
                pyboy.load_state(f)

            for _ in range(5):
                pyboy.tick()

            screen = pyboy.screen.image
            if screen.mode == 'RGBA':
                screen = screen.convert('RGB')

            debug_img = draw_debug_info(pyboy, screen, state_file)

            save_name = os.path.basename(state_file).replace(".state", ".jpg")
            save_path = os.path.join(OUTPUT_DIR, save_name)
            debug_img.save(save_path)

            print(f"✅ {save_name}")

        except Exception as e:
            print(f"⚠️ Erreur sur {state_file}: {e}")

    pyboy.stop()
    print(f"\n📂 Terminée ! Images dans '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()
