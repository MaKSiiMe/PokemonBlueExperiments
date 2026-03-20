"""
test_battle.py — Test manuel du BattleAgent.

Lance le jeu en SDL2. Tu joues manuellement jusqu'au combat.
Dès que 0xD057 > 0, le BattleAgent prend le relais automatiquement.

Contrôles (hors combat) :
    Z / Q / S / D  →  haut / gauche / bas / droite
    A              →  bouton A
    E              →  bouton B
    Echap          →  quitter

Usage :
    python test_battle.py
    python test_battle.py --state states/47_pewter_gym.state
"""

import argparse
import threading
import time
from pyboy import PyBoy
from pynput import keyboard as kb

from src.agent.battle_agent import BattleAgent
from src.emulator.pokemon_env import TICKS_PER_ACTION
from src.emulator.ram_map import RAM_BATTLE, RAM_FADING, RAM_BADGES

ROM_PATH = 'ROMs/PokemonBlue.gb'

KEY_MAP = {'z': 'up', 'q': 'left', 's': 'down', 'd': 'right', 'a': 'a', 'e': 'b'}

_stop        = threading.Event()
_manual_mode = threading.Event()
_manual_mode.set()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state', default='states/47_pewter_gym.state')
    p.add_argument('--speed', type=int, default=1)
    args = p.parse_args()

    pyboy = PyBoy(ROM_PATH, window='SDL2', sound=False, no_input=True)
    pyboy.set_emulation_speed(args.speed)

    with open(args.state, 'rb') as f:
        pyboy.load_state(f)
    for _ in range(60):
        pyboy.tick()

    held = set()

    def on_press(key):
        if not _manual_mode.is_set():
            return
        try:
            char = key.char.lower() if hasattr(key, 'char') and key.char else None
            if char in KEY_MAP and char not in held:
                held.add(char)
                pyboy.button_press(KEY_MAP[char])
        except Exception:
            pass

    def on_release(key):
        if key == kb.Key.esc:
            _stop.set()
            return False
        try:
            char = key.char.lower() if hasattr(key, 'char') and key.char else None
            if char in KEY_MAP and char in held:
                held.discard(char)
                pyboy.button_release(KEY_MAP[char])
        except Exception:
            pass

    def start_listener():
        time.sleep(1.0)   # délai anti-touches fantômes au démarrage
        listener = kb.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        listener.join()

    threading.Thread(target=start_listener, daemon=True).start()

    battle_agent = BattleAgent()
    prev_battle  = 0

    print(f"[Test] State : {args.state}")
    print("[Test] Z/Q/S/D=directions  A=bouton A  E=bouton B  Echap=quitter")
    print("[Test] BattleAgent actif automatiquement en combat.\n")

    try:
        while not _stop.is_set():
            battle = pyboy.memory[RAM_BATTLE]
            fading = pyboy.memory[RAM_FADING]

            if battle > 0 and prev_battle == 0:
                kind = 'sauvage' if battle == 1 else 'dresseur'
                print(f"[Test] Combat {kind} — BattleAgent actif.")
                _manual_mode.clear()
                battle_agent.reset()

            elif battle == 0 and prev_battle > 0:
                badges = bin(pyboy.memory[RAM_BADGES]).count('1')
                print(f"[Test] Combat terminé. Badges : {badges}/8")
                for btn in list(held):
                    pyboy.button_release(KEY_MAP[btn])
                held.clear()
                _manual_mode.set()

            prev_battle = battle

            if fading:
                pyboy.tick()
            elif battle > 0:
                action = battle_agent.act(pyboy)
                if action:
                    pyboy.button(action)
                for _ in range(TICKS_PER_ACTION):
                    pyboy.tick()
            else:
                pyboy.tick()
                pyboy.tick()

    except KeyboardInterrupt:
        pass
    finally:
        pyboy.stop()
        print("[Test] Arrêt.")


if __name__ == '__main__':
    main()
