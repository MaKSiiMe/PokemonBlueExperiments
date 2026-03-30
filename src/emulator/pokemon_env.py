"""
PokemonBlueEnv — Environnement Gymnasium wrappant PyBoy.

Observations et récompenses dérivées exclusivement de la RAM (pas de vision).

Observation vector (9 floats, tous dans [0, 1]) :
  0  player_x      — position X en tiles  (0xD362 / 255)
  1  player_y      — position Y en tiles  (0xD361 / 255)
  2  map_id        — ID de la map courante (0xD35E / 255)
  3  direction     — orientation          (0=bas 0.33=haut 0.66=gauche 1=droite)
  4  hp_pct        — HP joueur / max HP   (clampé [0, 1])
  5  battle_status — 0=overworld 0.5=sauvage 1.0=dresseur
  6  waypoint_x    — X cible si même map  (0 sinon)
  7  waypoint_y    — Y cible si même map  (0 sinon)
  8  badges_pct    — badges obtenus / 8

Action space (Discrete 6) :
  0=haut  1=bas  2=gauche  3=droite  4=a  5=b

Waypoint formats :
  Single  : waypoint=(map_id, x, y)
  Chaîné  : waypoints=[(map, x, y), ...]   — avance au suivant sans terminer
"""

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy

from src.emulator.ram_map import (
    RAM_PLAYER_X, RAM_PLAYER_Y, RAM_MAP_ID, RAM_DIRECTION,
    RAM_BATTLE, RAM_FADING, RAM_TEXT_ACTIVE,
    RAM_PLAYER_HP_H, RAM_PLAYER_HP_L, RAM_PLAYER_MHP_H, RAM_PLAYER_MHP_L,
    RAM_BADGES, RAM_ENEMY_LEVEL, RAM_EVENT_FLAGS, RAM_EVENT_LEN,
)

# Récompense par map (remplace le +1.0 générique pour les maps clés)
MAP_BONUSES: dict[int, float] = {
    0x28: 3.0,   # Labo Prof Chen
    0x36: 3.0,   # Arène de Brock
    0x25: 0.3,   # Maison 1F (peu d'intérêt d'y revenir)
}

ACTIONS          = ['up', 'down', 'left', 'right', 'a', 'b']
TICKS_PER_ACTION = 24   # ~0.4s à 60fps — assez long pour enregistrer un mouvement Gen 1

WAYPOINT_REACH_RADIUS = 1   # tiles — waypoint atteint dans ce rayon

_DIRECTION_MAP = {0x00: 0.0, 0x04: 0.33, 0x08: 0.66, 0x0C: 1.0}


class PokemonBlueEnv(gym.Env):
    """Environnement Gymnasium pour Pokémon Bleu — observations RAM uniquement."""

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        rom_path: str,
        init_state: str = 'states/00_pallet_town.state',
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

        if waypoints:
            self._waypoints = list(waypoints)
        elif waypoint:
            self._waypoints = [waypoint]
        else:
            self._waypoints = []
        self._wp_idx = 0

        window = 'null' if headless else 'SDL2'
        self.pyboy = PyBoy(rom_path, window=window, sound=False)
        self.pyboy.set_emulation_speed(speed)

        self.action_space      = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)

        self._step_count   = 0
        self._steps_stuck  = 0
        self._prev_map_id  = 0
        self._prev_x       = 0
        self._prev_y       = 0
        self._prev_hp      = 0
        self._prev_battle  = 0
        self._max_opp_lvl  = 0
        self._prev_badges  = 0
        self._prev_events  = 0
        self._visited_maps: set[int]         = set()
        self._tile_visits:  dict[tuple, int] = {}
        self._seen_tiles:   set[tuple]       = set()

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options:
            if 'waypoint' in options:
                self._waypoints = [options['waypoint']]
            if 'waypoints' in options:
                self._waypoints = list(options['waypoints'])

        self._wp_idx = 0
        self._load_state()

        self._step_count   = 0
        self._steps_stuck  = 0
        self._prev_x       = self._r(RAM_PLAYER_X)
        self._prev_y       = self._r(RAM_PLAYER_Y)
        self._prev_map_id  = self._r(RAM_MAP_ID)
        self._prev_hp      = self._r16(RAM_PLAYER_HP_H)
        self._prev_battle  = self._r(RAM_BATTLE)
        self._max_opp_lvl  = 0
        self._prev_badges  = self._r(RAM_BADGES)
        self._prev_events  = self._count_event_flags()
        self._visited_maps = {self._prev_map_id}
        self._tile_visits  = {}
        self._seen_tiles   = set()

        return self._observe(), {}

    def step(self, action_idx: int):
        action = ACTIONS[action_idx]
        if action in ('up', 'down', 'left', 'right'):
            self.pyboy.button_press(action)
            for _ in range(TICKS_PER_ACTION):
                self.pyboy.tick()
            self.pyboy.button_release(action)
        else:
            self.pyboy.button(action)
            for _ in range(TICKS_PER_ACTION):
                self.pyboy.tick()

        x   = self._r(RAM_PLAYER_X)
        y   = self._r(RAM_PLAYER_Y)
        mid = self._r(RAM_MAP_ID)

        reward = self._reward(x, y, mid)

        if self._wp_reached(x, y, mid):
            if self._wp_idx < len(self._waypoints) - 1:
                reward       += 2.0
                self._wp_idx += 1
                terminated    = False
            else:
                terminated = True
        elif self._blacked_out():
            terminated = True
        else:
            terminated = self._r(RAM_BADGES) & 0x01 > 0   # Badge Pierre obtenu

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
        return (self._r(addr) << 8) | self._r(addr + 1)

    def _load_state(self):
        if self.init_state and os.path.exists(self.init_state):
            with open(self.init_state, 'rb') as f:
                self.pyboy.load_state(f)
            for _ in range(60):
                self.pyboy.tick()
        else:
            print(f"[Env] State introuvable : {self.init_state}")

    @property
    def _waypoint(self) -> tuple[int, int, int] | None:
        if self._waypoints and self._wp_idx < len(self._waypoints):
            return self._waypoints[self._wp_idx]
        return None

    def _observe(self) -> np.ndarray:
        hp_max = max(self._r16(RAM_PLAYER_MHP_H), 1)
        hp_pct = min(self._r16(RAM_PLAYER_HP_H) / hp_max, 1.0)

        wp_x_norm = wp_y_norm = 0.0
        wp = self._waypoint
        if wp and self._r(RAM_MAP_ID) == wp[0]:
            wp_x_norm = wp[1] / 255.0
            wp_y_norm = wp[2] / 255.0

        return np.array([
            self._r(RAM_PLAYER_X)  / 255.0,
            self._r(RAM_PLAYER_Y)  / 255.0,
            self._r(RAM_MAP_ID)    / 255.0,
            _DIRECTION_MAP.get(self._r(RAM_DIRECTION), 0.0),
            hp_pct,
            self._r(RAM_BATTLE) / 2.0,
            wp_x_norm,
            wp_y_norm,
            bin(self._r(RAM_BADGES)).count('1') / 8.0,
        ], dtype=np.float32)

    def _reward(self, x: int, y: int, map_id: int) -> float:
        reward = -0.01
        battle = self._r(RAM_BATTLE)
        hp     = self._r16(RAM_PLAYER_HP_H)

        # Nouvelle map
        if map_id != self._prev_map_id:
            if map_id not in self._visited_maps:
                reward += MAP_BONUSES.get(map_id, 1.0)
            self._visited_maps.add(map_id)

        # Nouvelle tile
        tile = (map_id, x, y)
        if tile not in self._seen_tiles:
            self._seen_tiles.add(tile)
            reward += 0.5

        # Pénalité boucle (même tile > 600 fois)
        visits = self._tile_visits.get(tile, 0) + 1
        self._tile_visits[tile] = visits
        if visits > 600:
            reward -= 0.05

        # Mort en combat
        if hp == 0 and self._prev_hp > 0 and battle > 0:
            reward -= 1.0

        # Récompense fin de combat (niveau ennemi)
        if self._prev_battle > 0 and battle == 0:
            if self._max_opp_lvl > 0:
                reward += self._max_opp_lvl * 0.2
            self._max_opp_lvl = 0
        elif battle > 0:
            opp_lvl = self._r(RAM_ENEMY_LEVEL)
            if opp_lvl > self._max_opp_lvl:
                self._max_opp_lvl = opp_lvl

        # Nouveau badge (+50 par badge)
        badges = self._r(RAM_BADGES)
        new_badges = bin(badges).count('1') - bin(self._prev_badges).count('1')
        if new_badges > 0:
            reward += 50.0 * new_badges
        self._prev_badges = badges

        # Nouveau event flag — dresseur battu / event déclenché (+2 par flag)
        events = self._count_event_flags()
        if events > self._prev_events:
            reward += 2.0 * (events - self._prev_events)
        self._prev_events = events

        self._prev_hp     = hp
        self._prev_battle = battle

        # Distance shaping vers le waypoint courant
        wp = self._waypoint
        if wp and map_id == wp[0]:
            prev_dist = abs(self._prev_x - wp[1]) + abs(self._prev_y - wp[2])
            curr_dist = abs(x - wp[1])            + abs(y - wp[2])
            reward   += (prev_dist - curr_dist) * 0.1

        return reward

    def _count_event_flags(self) -> int:
        """Nombre de bits à 1 dans la zone event flags (0xD747, 32 octets)."""
        return sum(bin(self._r(RAM_EVENT_FLAGS + i)).count('1') for i in range(RAM_EVENT_LEN))

    def _blacked_out(self) -> bool:
        max_hp = self._r16(RAM_PLAYER_MHP_H)
        return max_hp > 0 and self._r16(RAM_PLAYER_HP_H) == 0 and self._r(RAM_BATTLE) == 0

    def _wp_reached(self, x: int, y: int, map_id: int) -> bool:
        wp = self._waypoint
        if not wp:
            return False
        return (map_id == wp[0]
                and abs(x - wp[1]) <= WAYPOINT_REACH_RADIUS
                and abs(y - wp[2]) <= WAYPOINT_REACH_RADIUS)
