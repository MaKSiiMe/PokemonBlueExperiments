from pyboy import PyBoy

ROM_PATH = "PokemonBlue.gb"


def main():
    pyboy = PyBoy(ROM_PATH, window="SDL2")
    pyboy.set_emulation_speed(1)

    print("🕵️  Lancement du scanner RAM...")
    print("👉 Déplace le joueur avec les flèches pour voir les valeurs !")

    ADDR_X_POS = 0xD362
    ADDR_Y_POS = 0xD361
    ADDR_MAP_N = 0xD35E

    with pyboy:
        while pyboy.tick():
            x_pos = pyboy.memory[ADDR_X_POS]
            y_pos = pyboy.memory[ADDR_Y_POS]
            map_id = pyboy.memory[ADDR_MAP_N]

            print(f"📍 Position: ({x_pos}, {y_pos}) | "
                  f"🗺️  Map ID: {map_id}   ", end="\r")


if __name__ == "__main__":
    main()
