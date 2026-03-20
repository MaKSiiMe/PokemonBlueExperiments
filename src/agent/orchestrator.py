"""
Orchestrator — Route vers le bon sous-agent selon l'état RAM.

Routing (0xD057) :
  0  → ExplorationAgent  (overworld)
  1  → BattleAgent       (combat sauvage)
  2  → BattleAgent       (combat dresseur)
  *  → press A           (dialogs, menus)

Transitions (0xD13F != 0) → tick sans action.
"""

from pyboy import PyBoy
from src.emulator.ram_map import RAM_BATTLE, RAM_FADING, RAM_TEXT_ACTIVE, RAM_MENU
from src.emulator.pokemon_env import TICKS_PER_ACTION, ACTIONS


class GameState:
    FADING         = 'fading'
    DIALOG         = 'dialog'
    OVERWORLD      = 'overworld'
    BATTLE_WILD    = 'battle_wild'
    BATTLE_TRAINER = 'battle_trainer'
    UNKNOWN        = 'unknown'


class Orchestrator:
    """
    Lit l'état du jeu depuis la RAM à chaque step et délègue au bon agent.

    Usage :
        orch = Orchestrator(pyboy, exploration_agent, battle_agent)
        while True:
            state = orch.step(obs)
    """

    def __init__(self, pyboy: PyBoy, exploration_agent, battle_agent):
        self.pyboy       = pyboy
        self.exploration = exploration_agent
        self.battle      = battle_agent
        self._prev_state = None

    def get_game_state(self) -> str:
        if self.pyboy.memory[RAM_FADING] != 0:
            return GameState.FADING

        battle = self.pyboy.memory[RAM_BATTLE]
        if battle == 1:
            return GameState.BATTLE_WILD
        if battle == 2:
            return GameState.BATTLE_TRAINER
        if self.pyboy.memory[RAM_TEXT_ACTIVE] != 0 or self.pyboy.memory[RAM_MENU] != 0:
            return GameState.DIALOG
        return GameState.OVERWORLD

    def step(self, obs) -> str:
        state = self.get_game_state()

        if state == GameState.FADING:
            for _ in range(TICKS_PER_ACTION):
                self.pyboy.tick()

        elif state == GameState.DIALOG:
            self.pyboy.button('a')
            for _ in range(TICKS_PER_ACTION):
                self.pyboy.tick()

        elif state in (GameState.BATTLE_WILD, GameState.BATTLE_TRAINER):
            btn = self.battle.act(self.pyboy)
            if btn:
                self.pyboy.button(btn)
            for _ in range(TICKS_PER_ACTION):
                self.pyboy.tick()

        elif state == GameState.OVERWORLD:
            action = self.exploration.act(obs)
            if action is not None:
                btn = ACTIONS[action]
                if btn in ('up', 'down', 'left', 'right'):
                    self.pyboy.button_press(btn)
                    for _ in range(TICKS_PER_ACTION):
                        self.pyboy.tick()
                    self.pyboy.button_release(btn)
                else:
                    self.pyboy.button(btn)
                    for _ in range(TICKS_PER_ACTION):
                        self.pyboy.tick()
            else:
                for _ in range(TICKS_PER_ACTION):
                    self.pyboy.tick()

        else:
            self.pyboy.button('b')
            for _ in range(TICKS_PER_ACTION):
                self.pyboy.tick()

        if state != self._prev_state:
            print(f"[Orchestrator] {self._prev_state} → {state}")
        self._prev_state = state
        return state
