"""
PokemonBlueEnv — Gymnasium environment wrapping PyBoy.

Observations and rewards are derived purely from RAM addresses.
No vision / YOLO required.

Observation vector (9 floats, all in [0, 1]):
  0  player_x      — X position in world tiles  (0xD362 / 255)
  1  player_y      — Y position in world tiles  (0xD361 / 255)
  2  map_id        — Current map ID             (0xD35E / 255)
  3  direction     — Facing direction           (0=down 0.33=up 0.66=left 1=right)
  4  hp_pct        — Player HP / max HP         (clamped [0, 1])
  5  battle_status — 0=overworld 0.5=wild 1.0=trainer
  6  waypoint_x    — Target X on waypoint map   (0xNN / 255, else 0.0)
  7  waypoint_y    — Target Y on waypoint map   (0xNN / 255, else 0.0)
  8  badges_pct    — Badges obtenus / 8         (progression globale)

Action space (Discrete 6):
  0=up  1=down  2=left  3=right  4=a  5=b

Waypoint formats:
  Single : waypoint=(map_id, x, y)         — termine quand atteint
  Chaîné : waypoints=[(map, x, y), ...]    — avance au suivant sans terminer,
                                              termine quand tous atteints
"""

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy

# ─── RAM addresses (source: pret/pokered + docs/ram_map.md) ──────────────────
RAM_PLAYER_X     = 0xD362   # Player X (world tiles)
RAM_PLAYER_Y     = 0xD361   # Player Y (world tiles)
RAM_MAP_ID       = 0xD35E   # Current map ID
RAM_DIRECTION    = 0xD35D   # Facing: 0x00=down 0x04=up 0x08=left 0x0C=right
RAM_BATTLE       = 0xD057   # Battle type: 0=overworld 1=wild 2=trainer
RAM_FADING       = 0xD13F   # Screen transition (non-zero = fading)
RAM_TEXT_ACTIVE  = 0xD11C   # Dialog active (non-zero)
RAM_PLAYER_HP_H  = 0xD16C   # Player current HP high byte
RAM_PLAYER_HP_L  = 0xD16D   # Player current HP low byte
RAM_PLAYER_MHP_H = 0xD18C   # Player max HP high byte
RAM_PLAYER_MHP_L = 0xD18D   # Player max HP low byte
RAM_BADGES       = 0xD356   # Badge bitmask
RAM_ENEMY_LVL    = 0xD018   # Niveau du Pokémon ennemi (pendant un combat)

BADGE_BOULDER    = 0x01     # Bit 0 = Badge Pierre (Brock)

ACTIONS = ['up', 'down', 'left', 'right', 'a', 'b']
TICKS_PER_ACTION = 24   # ~0.4s of game time at 60fps

_DIRECTION_MAP = {0x00: 0.0, 0x04: 0.33, 0x08: 0.66, 0x0C: 1.0}

WAYPOINT_REACH_RADIUS = 1   # tiles — waypoint considered reached within this radius


class PokemonBlueEnv(gym.Env):
    """Gymnasium environment for Pokémon Blue — RAM-only observations."""

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        rom_path: str,
        init_state: str = "states/06_pallet_town.state",
        headless: bool = True,
        speed: int = 0,
        max_steps: int = 10_000,
        waypoint:  tuple[int, int, int] | None = None,
        waypoints: list[tuple[int, int, int]] | None = None,
    ):
        super().__init__()

        self.rom_path   = rom_path
        self.init_state = init_state
        self.max_steps  = max_steps

        # Normalise : toujours travailler avec une liste interne
        if waypoints:
            self._waypoints = list(waypoints)
        elif waypoint:
            self._waypoints = [waypoint]
        else:
            self._waypoints = []
        self._wp_idx = 0

        window = "null" if headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window, sound=False)
        self.pyboy.set_emulation_speed(speed)

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(9,), dtype=np.float32
        )

        self._step_count   = 0
        self._steps_stuck  = 0
        self._prev_map_id  = 0
        self._prev_x       = 0
        self._prev_y       = 0
        self._prev_hp      = 0
        self._prev_battle  = 0
        self._max_opp_lvl  = 0
        self._visited_maps: set[int]          = set()
        self._tile_visits:  dict[tuple, int]  = {}

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options:
            if 'waypoint' in options:
                self._waypoints = [options['waypoint']]
            if 'waypoints' in options:
                self._waypoints = list(options['waypoints'])

        self._wp_idx       = 0
        self._load_state()
        self._step_count   = 0
        self._steps_stuck  = 0
        self._prev_x       = self._r(RAM_PLAYER_X)
        self._prev_y       = self._r(RAM_PLAYER_Y)
        self._prev_map_id  = self._r(RAM_MAP_ID)
        self._prev_hp      = self._r16(RAM_PLAYER_HP_H)
        self._prev_battle  = self._r(RAM_BATTLE)
        self._max_opp_lvl  = 0
        self._visited_maps = {self._prev_map_id}
        self._tile_visits  = {}
        return self._observe(), {}

    def step(self, action_idx: int):
        action = ACTIONS[action_idx]
        if action in ('up', 'down', 'left', 'right'):
            # Gen 1 requires the button to be held for ~16 ticks for movement to register
            self.pyboy.button_press(action)
            for _ in range(TICKS_PER_ACTION):
                self.pyboy.tick()
            self.pyboy.button_release(action)
        else:
            # 'a', 'b' — quick press is sufficient for menu/dialog interaction
            self.pyboy.button(action)
            for _ in range(TICKS_PER_ACTION):
                self.pyboy.tick()

        x   = self._r(RAM_PLAYER_X)
        y   = self._r(RAM_PLAYER_Y)
        mid = self._r(RAM_MAP_ID)

        reward = self._reward(x, y, mid)

        # Waypoint intermédiaire atteint → avancer sans terminer
        if self._wp_reached(x, y, mid):
            if self._wp_idx < len(self._waypoints) - 1:
                reward      += 2.0   # bonus de franchissement
                self._wp_idx += 1
                terminated   = False
            else:
                terminated   = True  # dernier waypoint atteint
        elif self._blacked_out():
            terminated = True        # black out → fin d'épisode
        else:
            terminated = self._r(RAM_BADGES) & BADGE_BOULDER > 0

        moved = x != self._prev_x or y != self._prev_y
        self._steps_stuck = 0 if moved else self._steps_stuck + 1
        self._prev_x      = x
        self._prev_y      = y
        self._prev_map_id = mid
        self._step_count += 1

        truncated = self._step_count >= self.max_steps

        return self._observe(), reward, terminated, truncated, {
            'map_id':      mid,
            'player_x':    x,
            'player_y':    y,
            'wp_idx':      self._wp_idx,
            'steps_stuck': self._steps_stuck,
        }

    def render(self):
        pass

    def close(self):
        self.pyboy.stop()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _r(self, addr: int) -> int:
        return self.pyboy.memory[addr]

    def _r16(self, addr: int) -> int:
        """Read a big-endian uint16 (Gen 1 convention)."""
        return (self._r(addr) << 8) | self._r(addr + 1)

    def _load_state(self):
        if self.init_state and os.path.exists(self.init_state):
            with open(self.init_state, 'rb') as f:
                self.pyboy.load_state(f)
            # 60 ticks needed for sprites to initialise (mov_stat > 0)
            # and for any fade/transition animation to complete
            for _ in range(60):
                self.pyboy.tick()
        else:
            print(f"[Env] State not found: {self.init_state}")

    @property
    def _waypoint(self) -> tuple[int, int, int] | None:
        """Waypoint courant (compatibilité avec l'ancien code)."""
        if self._waypoints and self._wp_idx < len(self._waypoints):
            return self._waypoints[self._wp_idx]
        return None

    def _observe(self) -> np.ndarray:
        hp_max = max(self._r16(RAM_PLAYER_MHP_H), 1)
        hp_pct = min(self._r16(RAM_PLAYER_HP_H) / hp_max, 1.0)

        wp_x_norm = 0.0
        wp_y_norm = 0.0
        wp = self._waypoint
        if wp:
            wp_map, wp_x, wp_y = wp
            if self._r(RAM_MAP_ID) == wp_map:
                wp_x_norm = wp_x / 255.0
                wp_y_norm = wp_y / 255.0

        badges_pct = bin(self._r(RAM_BADGES)).count('1') / 8.0

        return np.array([
            self._r(RAM_PLAYER_X)  / 255.0,
            self._r(RAM_PLAYER_Y)  / 255.0,
            self._r(RAM_MAP_ID)    / 255.0,
            _DIRECTION_MAP.get(self._r(RAM_DIRECTION), 0.0),
            hp_pct,
            self._r(RAM_BATTLE) / 2.0,
            wp_x_norm,
            wp_y_norm,
            badges_pct,
        ], dtype=np.float32)

    def _reward(self, x: int, y: int, map_id: int) -> float:
        reward  = -0.01
        battle  = self._r(RAM_BATTLE)
        hp      = self._r16(RAM_PLAYER_HP_H)

        # Zone change bonus — seulement pour les nouvelles maps
        if map_id != self._prev_map_id:
            if map_id not in self._visited_maps:
                reward += 1.0
            self._visited_maps.add(map_id)

        # Stuck penalty — tile visit counts (seuil 600 visites, même case)
        tile = (x, y, map_id)
        visits = self._tile_visits.get(tile, 0) + 1
        self._tile_visits[tile] = visits
        if visits > 600:
            reward -= 0.05

        # Death penalty — HP tombe à 0 pendant un combat
        if hp == 0 and self._prev_hp > 0 and battle > 0:
            reward -= 1.0

        # Opponent level reward — quand un combat se termine
        if self._prev_battle > 0 and battle == 0:
            if self._max_opp_lvl > 0:
                reward += self._max_opp_lvl * 0.2
            self._max_opp_lvl = 0
        elif battle > 0:
            opp_lvl = self._r(RAM_ENEMY_LVL)
            if opp_lvl > self._max_opp_lvl:
                self._max_opp_lvl = opp_lvl

        self._prev_hp     = hp
        self._prev_battle = battle

        # Distance shaping vers le waypoint courant
        wp = self._waypoint
        if wp:
            wp_map, wp_x, wp_y = wp
            if map_id == wp_map:
                prev_dist = abs(self._prev_x - wp_x) + abs(self._prev_y - wp_y)
                curr_dist = abs(x - wp_x)            + abs(y - wp_y)
                reward   += (prev_dist - curr_dist) * 0.1

        return reward

    def _blacked_out(self) -> bool:
        """HP=0 hors combat = black out (renvoyé au Centre Pokémon)."""
        return self._r16(RAM_PLAYER_HP_H) == 0 and self._r(RAM_BATTLE) == 0

    def _wp_reached(self, x: int, y: int, map_id: int) -> bool:
        wp = self._waypoint
        if not wp:
            return False
        wp_map, wp_x, wp_y = wp
        return (map_id == wp_map
                and abs(x - wp_x) <= WAYPOINT_REACH_RADIUS
                and abs(y - wp_y) <= WAYPOINT_REACH_RADIUS)
