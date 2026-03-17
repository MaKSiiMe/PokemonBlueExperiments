"""
Orchestrator — Routes to the correct sub-agent based on RAM game state.

Routing logic (0xD057):
  0  → Exploration agent  (overworld navigation)
  1  → Battle agent       (wild Pokémon)
  2  → Battle agent       (trainer battle)
  *  → Script handler     (dialogs, menus) — press A to skip

Screen transitions (0xD13F != 0) → tick without action.
"""

from pyboy import PyBoy

# ─── RAM addresses ────────────────────────────────────────────────────────────
RAM_BATTLE  = 0xD057   # 0=overworld  1=wild  2=trainer
RAM_FADING  = 0xD13F   # Non-zero during screen fade / zone transition
RAM_TEXT    = 0xD11C   # Non-zero when a dialog box is displayed
RAM_MENU    = 0xD12B   # Non-zero when a menu is open (start menu, bag, etc.)

TICKS_PER_ACTION = 24


class GameState:
    FADING         = 'fading'
    DIALOG         = 'dialog'
    OVERWORLD      = 'overworld'
    BATTLE_WILD    = 'battle_wild'
    BATTLE_TRAINER = 'battle_trainer'
    UNKNOWN        = 'unknown'


class Orchestrator:
    """
    Reads game state from RAM every step and delegates to the right agent.

    Usage:
        orch = Orchestrator(pyboy, exploration_agent, battle_agent)
        while True:
            obs  = env._observe()
            state = orch.step(obs)
    """

    def __init__(self, pyboy: PyBoy, exploration_agent, battle_agent):
        self.pyboy       = pyboy
        self.exploration = exploration_agent
        self.battle      = battle_agent
        self._prev_state = None
        self._step_count = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def get_game_state(self) -> str:
        if self.pyboy.memory[RAM_FADING] != 0:
            return GameState.FADING

        battle = self.pyboy.memory[RAM_BATTLE]
        if battle == 1:
            return GameState.BATTLE_WILD
        if battle == 2:
            return GameState.BATTLE_TRAINER
        if battle == 0:
            if self.pyboy.memory[RAM_TEXT] != 0 or self.pyboy.memory[RAM_MENU] != 0:
                return GameState.DIALOG
            return GameState.OVERWORLD

        return GameState.UNKNOWN

    def step(self, obs) -> str:
        """
        Choose and execute one action based on current game state.
        Returns the GameState string for logging / curriculum checks.
        """
        state = self.get_game_state()

        if state == GameState.FADING:
            # Let the transition animation play out — no input
            for _ in range(TICKS_PER_ACTION):
                self.pyboy.tick()

        elif state == GameState.DIALOG:
            # Auto-skip dialog / text boxes by pressing A
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
                btn = ['up', 'down', 'left', 'right', 'a', 'b'][action]
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
            # Unknown state (cutscene, intro, etc.) — press B to dismiss
            self.pyboy.button('b')
            for _ in range(TICKS_PER_ACTION):
                self.pyboy.tick()

        if state != self._prev_state:
            print(f"[Orchestrator] State: {self._prev_state} → {state}")
        self._prev_state  = state
        self._step_count += 1
        return state
