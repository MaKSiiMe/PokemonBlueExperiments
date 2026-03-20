import sys
from pyboy import PyBoy

if len(sys.argv) < 2:
    sys.exit(1)

state_file = sys.argv[1]
rom_path = "ROMs/PokemonBlue.gb"

try:
    pyboy = PyBoy(rom_path, window="null", sound=False)
    pyboy.set_emulation_speed(0)

    with open(state_file, "rb") as f:
        pyboy.load_state(f)

    for _ in range(100):
        pyboy.tick()

    pyboy.stop()
    sys.exit(0)

except Exception:
    sys.exit(1)
