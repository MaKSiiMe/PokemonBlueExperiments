"""
Orchestrator — Route vers le bon sous-agent selon l'état RAM.

Routing (wIsInBattle) :
  0               → ExplorationAgent  (overworld, dialogues et menus compris)
  WILD_BATTLE     → BattleAgent       (combat sauvage)
  TRAINER_BATTLE  → BattleAgent       (combat dresseur)
  autre           → press B

Les états FADING et DIALOG reposaient sur des adresses fausses (wPrize3,
wCapturedMonSpecies, wLinkState) et ont été retirés : la détection de mode,
validée sur savestates étiquetés, arrive en Phase 1 avec un orchestrateur réécrit.
"""

from pyboy import PyBoy

from pokeblue.state import ram_symbols as sym
from src.emulator.pokemon_env import ACTIONS, TICKS_PER_ACTION
from src.emulator.ram_map import RAM_BATTLE


class GameState:
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
        battle = self.pyboy.memory[RAM_BATTLE]
        if battle == 0:
            return GameState.OVERWORLD
        if battle == sym.WILD_BATTLE:
            return GameState.BATTLE_WILD
        if battle == sym.TRAINER_BATTLE:
            return GameState.BATTLE_TRAINER
        return GameState.UNKNOWN

    def step(self, obs) -> str:
        state = self.get_game_state()

        if state in (GameState.BATTLE_WILD, GameState.BATTLE_TRAINER):
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
