"""
PokemonBlueEnv — Environnement Gymnasium wrappant PyBoy.

Point de départ : labo du Prof. Chen, juste après avoir reçu le Pokédex
(parcel livré, starter obtenu). Objectif : battre Brock (Badge Pierre).

Espace d'observation hybride (Dict) :
  ── screen       (3, 72, 80)  float32 [0, 1] ─────────────────────────────────
      3 frames grisées empilées (frame stacking temporel).
      Chaque frame : écran Game Boy (144×160) sous-échantillonné ×2 → 72×80,
      converti en niveaux de gris, normalisé dans [0, 1].

  ── visited_mask (1, 48, 48)  float32 {0, 1} ─────────────────────────────────
      Grille binaire 48×48 centrée sur la position du joueur.
      1 = tuile (map_id, x, y) déjà visitée dans l'épisode courant (ou antérieur).
      0 = inconnue.

  ── ram          (16,)        float32 [0, 1] ─────────────────────────────────
      0  player_x          — position X en tiles  (0xD362 / 255)
      1  player_y          — position Y en tiles  (0xD361 / 255)
      2  map_id            — ID de la map courante (0xD35E / 255)
      3  direction         — orientation          (0=bas 0.33=haut 0.66=gauche 1=droite)
      4  hp_pct            — HP joueur / max HP   (clampé [0, 1])
      5  battle_status     — 0=overworld 0.5=sauvage 1.0=dresseur
      6  event_flags_pct   — flags événement activés / total (progression globale)
      7  steps_stuck_norm  — steps bloqué / 100 (clampé [0, 1], signal d'urgence)
      8  badges_pct        — badges obtenus / 8
      9  type_advantage    — meilleur multiplicateur dispo / 4.0
      10 enemy_can_evolve  — 1.0 si l'ennemi a une évolution, 0.0 sinon
      11 zone_density      — Pokémon rencontrables dans la zone / 8.0
      12 battle_mon_hp_pct — HP du Pokémon actif en combat / HP max (D015/D023)
      13 pokedex_pct       — espèces capturées / 151 (D2F7, masque de bits)
      14 money_norm        — argent BCD décodé / 999999 (D347-D349)
      15 items_norm        — objets uniques dans le sac / 20 (CF7B)

Action space (Discrete 7) :
  0=haut  1=bas  2=gauche  3=droite  4=a  5=b  6=start
"""

import os
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy

from src.emulator.ram_map import (
    RAM_PLAYER_X, RAM_PLAYER_Y, RAM_MAP_ID, RAM_DIRECTION,
    RAM_BATTLE, RAM_FADING, RAM_TEXT_ACTIVE,
    RAM_PLAYER_HP_H, RAM_PLAYER_HP_L, RAM_PLAYER_MHP_H, RAM_PLAYER_MHP_L,
    RAM_BADGES, RAM_ENEMY_LEVEL, RAM_EVENT_FLAGS, RAM_EVENT_LEN,
    RAM_ENEMY_TYPE1, RAM_ENEMY_TYPE2, RAM_ENEMY_SPECIES,
    RAM_MOVE_IDS, RAM_MOVE_PP,
    RAM_PARTY_COUNT, RAM_PARTY_LEVELS, RAM_PARTY_HP, RAM_PARTY_MAX_HP,
    RAM_BATTLE_MON_HP_H, RAM_BATTLE_MON_MAX_HP_H,
    RAM_ITEM_COUNT,
    RAM_MONEY,
    RAM_POKEDEX_OWNED, RAM_POKEDEX_LEN, RAM_POKEDEX_MAX,
)
from src.knowledge.gen1_data import (
    RAM_TYPE_BYTE_TO_NAME, TYPE_CHART,
    GEN1_INTERNAL_TO_DEX, MOVE_TYPES, STATUS_MOVES,
)
from src.knowledge import PokemonKnowledgeGraph

# ── Constantes d'observation ──────────────────────────────────────────────────
SCREEN_H     = 72    # hauteur après sous-échantillonnage ×2 (144 → 72)
SCREEN_W     = 80    # largeur après sous-échantillonnage ×2 (160 → 80)
N_STACK      = 3     # nombre de frames empilées (frame stacking)
FRAME_SKIP   = 1     # répétitions de l'action avant d'observer (×1 SPS)
MASK_SIZE    = 48    # taille du masque de visite centré sur le joueur (48×48)
RAM_VEC_SIZE = 16    # taille du vecteur scalaire RAM

# Récompense par map (remplace le +3.0 générique pour les maps clés)
MAP_BONUSES: dict[int, float] = {
    0x28: 3.0,    # Labo Prof Chen
    0x25: 0.3,    # Maison 1F Bourg Palette (peu d'intérêt)
    0x0C: 20.0,   # Route 1 — frontière critique (×4)
    0x01: 30.0,   # Bourg des Eaux — premier vrai jalon (×4)
    0x0D: 20.0,   # Route 2
    0x33: 20.0,   # Forêt Viridian
    0x02: 30.0,   # Argenta — ville de Brock
    0x36: 50.0,   # Arène Brock — objectif final
}

ACTIONS          = ['up', 'down', 'left', 'right', 'a', 'b', 'start']
TICKS_PER_ACTION = 24   # ~0.4s à 60fps — durée d'une animation de déplacement Gen 1

_DIRECTION_MAP = {0x00: 0.0, 0x04: 0.33, 0x08: 0.66, 0x0C: 1.0}


class PokemonBlueEnv(gym.Env):
    """Environnement Gymnasium pour Pokémon Bleu — espace d'observation hybride Dict."""

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        rom_path: str,
        init_state: str = 'states/00_pallet_town.state',
        headless: bool = True,
        speed: int = 0,
        max_steps: int = 10_000,
        kg: PokemonKnowledgeGraph | None = None,
    ):
        super().__init__()

        self.rom_path   = rom_path
        self.init_state = init_state
        self.max_steps  = max_steps

        # Graphe de connaissances — partagé entre envs parallèles pour éviter
        # de charger N fois le même fichier JSON.
        self._kg: PokemonKnowledgeGraph = kg or PokemonKnowledgeGraph()

        # Cache dex → can_evolve pour un lookup O(1) à chaque step
        self._dex_can_evolve: dict[int, bool] = {
            data["dex"]: bool(self._kg.evolutions(data["name"]))
            for _, data in self._kg._G.nodes(data=True)
            if data.get("kind") == "pokemon"
        }

        self._escaped_lab        = False
        self._min_y_progress     = 255
        self._entered_building   = False
        self._steps_on_current_map = 0

        # Chemin optimal Bourg Palette → Arène de Pierre (carte une fois pour toutes)
        _path = self._kg.zone_path(0x00, 0x36)
        self._optimal_path_zones: frozenset[int] = frozenset(_path)

        window = 'null' if headless else 'SDL2'
        self.pyboy = PyBoy(rom_path, window=window, sound=False)
        self.pyboy.set_emulation_speed(speed)

        # ── Espace d'observation hybride Dict ─────────────────────────────────
        self.observation_space = spaces.Dict({
            'screen': spaces.Box(
                low=0.0, high=1.0,
                shape=(N_STACK, SCREEN_H, SCREEN_W),
                dtype=np.float32,
            ),
            'visited_mask': spaces.Box(
                low=0.0, high=1.0,
                shape=(1, MASK_SIZE, MASK_SIZE),
                dtype=np.float32,
            ),
            'ram': spaces.Box(
                low=0.0, high=1.0,
                shape=(RAM_VEC_SIZE,),
                dtype=np.float32,
            ),
        })

        self.action_space = spaces.Discrete(len(ACTIONS))

        # ── Buffers d'état internes ────────────────────────────────────────────
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=N_STACK)

        self._step_count        = 0
        self._steps_stuck       = 0
        self._prev_map_id       = 0
        self._prev_x            = 0
        self._prev_y            = 0
        self._prev_hp           = 0
        self._prev_battle       = 0
        self._prev_badges       = 0
        self._prev_events       = 0
        self._prev_level_reward = 0.0
        self._prev_total_hp     = 0
        self._visited_maps:    set[int]         = set()
        self._tile_visits:     dict[tuple, int] = {}
        self._seen_tiles:      set[tuple]       = set()   # persiste entre épisodes
        # Grilles numpy par map (256×256 bool) pour _get_visited_mask() vectorisé
        self._seen_arrays:     dict[int, np.ndarray] = {}
        self._zone_density_cache: dict[int, float] = {}

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_state()

        # Initialise le buffer de frames avec N_STACK copies de la frame initiale
        first_frame = self._get_screen()
        self._frame_buffer.clear()
        for _ in range(N_STACK):
            self._frame_buffer.append(first_frame)

        self._step_count        = 0
        self._steps_stuck       = 0
        self._prev_x            = self._r(RAM_PLAYER_X)
        self._prev_y            = self._r(RAM_PLAYER_Y)
        self._prev_map_id       = self._r(RAM_MAP_ID)
        self._prev_hp           = self._r16(RAM_PLAYER_HP_H)
        self._prev_battle       = self._r(RAM_BATTLE)
        self._prev_badges       = self._r(RAM_BADGES)
        self._prev_events       = self._count_event_flags()
        self._prev_level_reward = self._r_level()
        self._prev_total_hp     = self._total_party_hp()
        self._visited_maps = {self._prev_map_id}
        self._tile_visits        = {}
        self._escaped_lab        = False
        self._min_y_progress     = 255
        self._entered_building   = False
        self._steps_on_current_map = 0
        # _seen_tiles intentionnellement NON resetté : persiste entre épisodes
        self._prev_move_pp = [self._r(RAM_MOVE_PP[i]) for i in range(4)]

        return self._observe(), {}

    def step(self, action_idx: int):
        action = ACTIONS[action_idx]

        if action in ('up', 'down', 'left', 'right'):
            self.pyboy.button_press(action)
            self.pyboy.tick(TICKS_PER_ACTION - 1, render=False)
            self.pyboy.tick(1, render=True)
            self.pyboy.button_release(action)
        else:
            self.pyboy.button(action)
            self.pyboy.tick(TICKS_PER_ACTION - 1, render=False)
            self.pyboy.tick(1, render=True)

        self._frame_buffer.append(self._get_screen())

        x   = self._r(RAM_PLAYER_X)
        y   = self._r(RAM_PLAYER_Y)
        mid = self._r(RAM_MAP_ID)

        reward = self._reward(x, y, mid)

        if self._blacked_out():
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

        return self._observe(), reward, terminated, truncated, self._info(x, y, mid)

    def render(self):
        pass

    def close(self):
        self.pyboy.stop()

    # ── Go-Explore API ────────────────────────────────────────────────────────

    def capture_state(self) -> bytes:
        """Sérialise l'état complet de l'émulateur en bytes.

        Utilisé par GoExploreWrapper pour archiver les cellules.

        Returns:
            Savestate PyBoy en mémoire (bytes).
        """
        import io
        buf = io.BytesIO()
        self.pyboy.save_state(buf)
        return buf.getvalue()

    def reset_from_state(self, savestate_bytes: bytes) -> tuple:
        """Réinitialise l'env en chargeant un savestate arbitraire (bytes).

        Utilisé par GoExploreWrapper pour "téléporter" l'agent vers une
        cellule de l'archive plutôt que de repartir de Bourg Palette.

        Args:
            savestate_bytes : état PyBoy précédemment capturé via capture_state().

        Returns:
            (observation, info) — même interface que reset().
        """
        import io
        buf = io.BytesIO(savestate_bytes)
        self.pyboy.load_state(buf)
        self.pyboy.tick(1, render=True)

        # Ré-initialise le frame buffer avec l'écran du savestate
        first_frame = self._get_screen()
        self._frame_buffer.clear()
        for _ in range(N_STACK):
            self._frame_buffer.append(first_frame)

        self._step_count        = 0
        self._steps_stuck       = 0
        self._prev_x            = self._r(RAM_PLAYER_X)
        self._prev_y            = self._r(RAM_PLAYER_Y)
        self._prev_map_id       = self._r(RAM_MAP_ID)
        self._prev_hp           = self._r16(RAM_PLAYER_HP_H)
        self._prev_battle       = self._r(RAM_BATTLE)
        self._prev_badges       = self._r(RAM_BADGES)
        self._prev_events       = self._count_event_flags()
        self._prev_level_reward = self._r_level()
        self._prev_total_hp     = self._total_party_hp()
        self._prev_move_pp      = [self._r(RAM_MOVE_PP[i]) for i in range(4)]
        self._tile_visits          = {}
        self._escaped_lab          = False
        self._min_y_progress       = 255
        self._entered_building     = False
        self._steps_on_current_map = 0
        self._visited_maps         = {self._prev_map_id}   # reset : chaque épisode redécouvre les maps
        # _seen_tiles conservé intentionnellement (connaissance globale persistante)

        return self._observe(), {}

    # ── Helpers d'observation ─────────────────────────────────────────────────

    def _get_screen(self) -> np.ndarray:
        """Extrait l'écran courant en niveaux de gris sous-échantillonné (72×80).

        Returns:
            ndarray float32 shape (72, 80), valeurs dans [0, 1].
        """
        # pyboy.screen.ndarray → (144, 160, 4) uint8 RGBA
        raw = self.pyboy.screen.ndarray  # (144, 160, 4)
        # Conversion niveaux de gris via pondération luminance standard
        gray = (
            0.299  * raw[:, :, 0].astype(np.float32) +
            0.587  * raw[:, :, 1].astype(np.float32) +
            0.114  * raw[:, :, 2].astype(np.float32)
        )  # (144, 160)
        # Sous-échantillonnage ×2 → (72, 80)
        downsampled = gray[::2, ::2]
        return (downsampled / 255.0).astype(np.float32)

    def _get_visited_mask(self, map_id: int, x: int, y: int) -> np.ndarray:
        """Construit la grille binaire 48×48 de visite centrée sur (x, y).

        Utilise un tableau numpy 256×256 par map pour un slicing O(1)
        au lieu d'une double boucle Python O(48²).

        Returns:
            ndarray float32 shape (1, 48, 48).
        """
        arr = self._seen_arrays.get(map_id)
        if arr is None:
            return np.zeros((1, MASK_SIZE, MASK_SIZE), dtype=np.float32)

        half = MASK_SIZE // 2
        x0, x1 = x - half, x + half   # indices dans la grille map
        y0, y1 = y - half, y + half

        # Indices clampés dans [0, 256)
        cx0, cx1 = max(x0, 0), min(x1, 256)
        cy0, cy1 = max(y0, 0), min(y1, 256)

        # Position du patch dans le masque 48×48
        mx0 = cx0 - x0
        my0 = cy0 - y0

        mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32)
        patch = arr[cx0:cx1, cy0:cy1]
        mask[mx0:mx0 + patch.shape[0], my0:my0 + patch.shape[1]] = patch
        return mask.reshape(1, MASK_SIZE, MASK_SIZE)

    def _observe(self) -> dict:
        """Construit l'observation hybride Dict à partir de l'état courant.

        Returns:
            dict avec les clés 'screen', 'visited_mask', 'ram'.
        """
        x      = self._r(RAM_PLAYER_X)
        y      = self._r(RAM_PLAYER_Y)
        map_id = self._r(RAM_MAP_ID)

        # ── screen : empilement des N_STACK frames (3, 72, 80) ────────────────
        screen_stack = np.stack(list(self._frame_buffer), axis=0)  # (3, 72, 80)

        # ── visited_mask : grille binaire 48×48 ───────────────────────────────
        visited_mask = self._get_visited_mask(map_id, x, y)        # (1, 48, 48)

        # ── ram : vecteur scalaire normalisé (12,) ────────────────────────────
        hp_max = max(self._r16(RAM_PLAYER_MHP_H), 1)
        hp_pct = min(self._r16(RAM_PLAYER_HP_H) / hp_max, 1.0)

        # Progression globale via event flags (remplace waypoint_x/y)
        self._cached_events = self._count_event_flags()
        event_flags_pct  = self._cached_events / max(RAM_EVENT_LEN * 8, 1)
        steps_stuck_norm = min(self._steps_stuck / 100.0, 1.0)

        type_advantage, enemy_can_evolve = self._kg_battle_signals()
        if map_id not in self._zone_density_cache:
            self._zone_density_cache[map_id] = min(
                len(self._kg.encounters_in_zone(map_id)) / 8.0, 1.0
            )
        zone_density = self._zone_density_cache[map_id]

        # ── Nouveaux signaux RAM ──────────────────────────────────────────────
        # HP du Pokémon actif en combat (D015/D023) — 0.0 en overworld
        battle_mon_hp_pct = 0.0
        if self._r(RAM_BATTLE) > 0:
            bm_max = max(self._r16(RAM_BATTLE_MON_MAX_HP_H), 1)
            battle_mon_hp_pct = min(self._r16(RAM_BATTLE_MON_HP_H) / bm_max, 1.0)

        # Pokédex : proportion d'espèces capturées / 151
        owned_count = sum(
            self._r(RAM_POKEDEX_OWNED + i).bit_count()
            for i in range(RAM_POKEDEX_LEN)
        )
        pokedex_pct = min(owned_count / RAM_POKEDEX_MAX, 1.0)

        # Argent BCD (D347-D349) normalisé sur 999 999
        money_norm = self._decode_bcd(
            self._r(RAM_MONEY[0]),
            self._r(RAM_MONEY[1]),
            self._r(RAM_MONEY[2]),
        ) / 999_999.0

        # Nombre d'objets uniques dans le sac (max 20)
        items_norm = min(self._r(RAM_ITEM_COUNT) / 20.0, 1.0)

        ram_vec = np.array([
            x                  / 255.0,
            y                  / 255.0,
            map_id             / 255.0,
            _DIRECTION_MAP.get(self._r(RAM_DIRECTION), 0.0),
            hp_pct,
            self._r(RAM_BATTLE) / 2.0,
            event_flags_pct,
            steps_stuck_norm,
            bin(self._r(RAM_BADGES)).count('1') / 8.0,
            type_advantage,
            enemy_can_evolve,
            zone_density,
            battle_mon_hp_pct,
            pokedex_pct,
            money_norm,
            items_norm,
        ], dtype=np.float32)

        return {
            'screen':       screen_stack,
            'visited_mask': visited_mask,
            'ram':          ram_vec,
        }

    # ── Helpers RAM ───────────────────────────────────────────────────────────

    def _r(self, addr: int) -> int:
        return self.pyboy.memory[addr]

    def _r16(self, addr: int) -> int:
        return (self._r(addr) << 8) | self._r(addr + 1)

    def _load_state(self):
        if self.init_state and os.path.exists(self.init_state):
            with open(self.init_state, 'rb') as f:
                self.pyboy.load_state(f)
            self.pyboy.tick(60, render=False)
        else:
            print(f"[Env] State introuvable : {self.init_state}")

    def _kg_battle_signals(self) -> tuple[float, float]:
        """Calcule les signaux KG liés au combat (indices 9 et 10).

        Returns:
            type_advantage   : meilleur multiplicateur dispo / 4.0  → [0, 1]
            enemy_can_evolve : 1.0 si l'ennemi a une évolution, 0.0 sinon

        En overworld (RAM_BATTLE == 0) : retourne (0.5, 0.0).
        """
        if self._r(RAM_BATTLE) == 0:
            return 0.5, 0.0

        enemy_type1 = self._r(RAM_ENEMY_TYPE1)
        enemy_type2 = self._r(RAM_ENEMY_TYPE2)
        enemy_types = [
            RAM_TYPE_BYTE_TO_NAME.get(enemy_type1, "normal"),
            RAM_TYPE_BYTE_TO_NAME.get(enemy_type2, "normal"),
        ]

        best_mult = 0.0
        for i in range(4):
            pp      = self._r(RAM_MOVE_PP[i])
            move_id = self._r(RAM_MOVE_IDS[i])
            if pp == 0 or move_id == 0:
                continue
            if move_id in STATUS_MOVES:
                continue
            move_type_byte = MOVE_TYPES.get(move_id, 0x00)
            move_type_name = RAM_TYPE_BYTE_TO_NAME.get(move_type_byte, "normal")
            mult = 1.0
            for def_type in enemy_types:
                mult *= TYPE_CHART.get(move_type_name, {}).get(def_type, 1.0)
            if mult > best_mult:
                best_mult = mult

        type_advantage = min(best_mult / 4.0, 1.0)

        internal_id      = self._r(RAM_ENEMY_SPECIES)
        dex_number       = GEN1_INTERNAL_TO_DEX.get(internal_id)
        enemy_can_evolve = 1.0 if self._dex_can_evolve.get(dex_number, False) else 0.0

        return type_advantage, enemy_can_evolve

    def _info(self, x: int, y: int, map_id: int) -> dict:
        """Construit le dictionnaire info retourné à chaque step.

        Contient les métriques de base (navigation) ainsi que les métriques
        de jeu agrégées par GameMetricsCallback pour le monitoring TensorBoard.

        Milestones encodés en bits (0/1) — lus depuis _visited_maps et RAM_BADGES :
          ms_viridian : Bourg des Eaux (map 0x01) visitée
          ms_forest   : Forêt de Jade  (map 0x33) visitée
          ms_pewter   : Argenta         (map 0x02) visitée
          ms_badge1   : Badge Pierre obtenu (bit 0 de RAM_BADGES)
          ms_mt_moon  : Mont Sélénite   (map 0x59) visitée
        """
        party_size = min(self._r(RAM_PARTY_COUNT), 6)
        max_level  = (
            max(self._r(RAM_PARTY_LEVELS[i]) for i in range(party_size))
            if party_size > 0 else 0
        )
        pokedex_owned = sum(
            self._r(RAM_POKEDEX_OWNED + i).bit_count()
            for i in range(RAM_POKEDEX_LEN)
        )
        badges = self._r(RAM_BADGES)

        return {
            # Navigation
            'map_id':       map_id,
            'player_x':     x,
            'player_y':     y,
            'steps_stuck':  self._steps_stuck,
            # Métriques du jeu
            'unique_maps':     len(self._visited_maps),
            'max_level':       max_level,
            'n_badges':        bin(badges).count('1'),
            'pokedex_owned':   pokedex_owned,
            'episode_steps':   self._step_count,
            # Milestones (0/1) — utilisés pour le taux de complétion
            'ms_viridian':  int(0x01 in self._visited_maps),
            'ms_forest':    int(0x33 in self._visited_maps),
            'ms_pewter':    int(0x02 in self._visited_maps),
            'ms_badge1':    int(bool(badges & 0x01)),
            'ms_mt_moon':   int(0x59 in self._visited_maps),
            # Progression géographique (pour diagnostic)
            'min_y_progress':         self._min_y_progress,
            'steps_on_current_map':   self._steps_on_current_map,
            # Composantes de reward (step courant)
            **getattr(self, '_r_components', {}),
        }

    def _reward(self, x: int, y: int, map_id: int) -> float:
        """Reward minimaliste — Test 1 : 4 signaux uniquement.

        Objectif : vérifier si la policy peut apprendre à progresser
        sans shaping complexe. Si unique_maps_ever_total > 6 à 500k steps
        avec cette version, les itérations de reward précédentes créaient
        des artéfacts bloquants. Sinon, le problème est dans l'architecture.
        """
        reward = 0.0
        r_map  = 0.0
        r_tile = 0.0
        r_event = 0.0

        # Nouvelle map jamais visitée dans cet épisode
        if map_id != self._prev_map_id:
            if map_id not in self._visited_maps:
                r_map = MAP_BONUSES.get(map_id, 10.0)
            self._visited_maps.add(map_id)

        # Nouvelle tile jamais vue dans toute la vie de l'agent
        tile = (map_id, x, y)
        if tile not in self._seen_tiles:
            r_tile = 1.0
            self._seen_tiles.add(tile)
            if map_id not in self._seen_arrays:
                self._seen_arrays[map_id] = np.zeros((256, 256), dtype=bool)
            self._seen_arrays[map_id][x, y] = True

        # Nouveau badge
        badges     = self._r(RAM_BADGES)
        new_badges = bin(badges).count('1') - bin(self._prev_badges).count('1')
        if new_badges > 0:
            r_event += 50.0 * new_badges
        self._prev_badges = badges

        # Nouveau event flag
        events = getattr(self, '_cached_events', self._count_event_flags())
        if events > self._prev_events:
            r_event += 2.0 * (events - self._prev_events)
        self._prev_events = events

        reward = r_map + r_tile + r_event
        self._r_components = {
            'r_map':   r_map,
            'r_tile':  r_tile,
            'r_event': r_event,
        }
        return reward

    def _type_intent_bonus(self) -> float:
        """Bonus si le move utilisé ce step est Super Efficace (×2.0 ou ×4.0).

        Détecte quel slot a perdu un PP depuis le step précédent → move joué.

        Returns:
            +0.1 si SE · +0.2 si double SE · 0.0 sinon ou si move de statut.
        """
        enemy_type1 = self._r(RAM_ENEMY_TYPE1)
        enemy_type2 = self._r(RAM_ENEMY_TYPE2)
        enemy_types = [
            RAM_TYPE_BYTE_TO_NAME.get(enemy_type1, "normal"),
            RAM_TYPE_BYTE_TO_NAME.get(enemy_type2, "normal"),
        ]

        bonus = 0.0
        for i in range(4):
            curr_pp = self._r(RAM_MOVE_PP[i])
            if curr_pp < self._prev_move_pp[i]:
                move_id = self._r(RAM_MOVE_IDS[i])
                if move_id and move_id not in STATUS_MOVES:
                    move_type_byte = MOVE_TYPES.get(move_id, 0x00)
                    move_type_name = RAM_TYPE_BYTE_TO_NAME.get(move_type_byte, "normal")
                    mult = 1.0
                    for def_type in enemy_types:
                        mult *= TYPE_CHART.get(move_type_name, {}).get(def_type, 1.0)
                    if mult >= 4.0:
                        bonus = 0.2
                    elif mult >= 2.0:
                        bonus = 0.1
            self._prev_move_pp[i] = curr_pp

        return bonus

    def _decode_bcd(self, *bytes_vals: int) -> int:
        """Décode une valeur Binary Coded Decimal (BCD) en entier décimal.

        Chaque octet encode deux chiffres décimaux (quartet haut / bas).
        Ex : _decode_bcd(0x09, 0x99, 0x99) → 99 999

        Args:
            *bytes_vals: octets du plus significatif au moins significatif.

        Returns:
            Valeur entière décodée.
        """
        result = 0
        for b in bytes_vals:
            result = result * 100 + (b >> 4) * 10 + (b & 0x0F)
        return result

    def _count_event_flags(self) -> int:
        """Nombre de bits à 1 dans la zone event flags (0xD747, 32 octets)."""
        return sum(self._r(RAM_EVENT_FLAGS + i).bit_count() for i in range(RAM_EVENT_LEN))

    def _total_party_hp(self) -> int:
        """Somme des HP courants de tous les Pokémon de l'équipe (hors combat)."""
        n = min(self._r(RAM_PARTY_COUNT), 6)
        return sum(self._r16(RAM_PARTY_HP[i]) for i in range(n))

    def _total_party_max_hp(self) -> int:
        """Somme des HP maximum de tous les Pokémon de l'équipe."""
        n = min(self._r(RAM_PARTY_COUNT), 6)
        return sum(self._r16(RAM_PARTY_MAX_HP[i]) for i in range(n))

    def _r_level(self) -> float:
        """Valeur de la fonction de récompense de niveau (formule affine par morceaux).

        Lit les niveaux des Pokémon présents dans l'équipe (wPartyCount slots)
        et applique :
            R = ∑ levels               si ∑ levels < 15
            R = 30 + (∑ levels - 15)/4  sinon

        La récompense injectée à chaque step est le delta R(t) - R(t-1),
        ce qui donne un signal positif à chaque gain de niveau et un rendement
        marginal décroissant au-delà du seuil 15 pour éviter le grinding.

        Returns:
            float — valeur courante de la fonction R_level.
        """
        party_size = min(self._r(RAM_PARTY_COUNT), 6)
        total = sum(self._r(RAM_PARTY_LEVELS[i]) for i in range(party_size))
        if total < 15:
            return float(total)
        return 30.0 + (total - 15) / 4.0

    def _blacked_out(self) -> bool:
        max_hp = self._r16(RAM_PLAYER_MHP_H)
        return max_hp > 0 and self._r16(RAM_PLAYER_HP_H) == 0 and self._r(RAM_BATTLE) == 0

    # ── Action Masking (requis par MaskablePPO / sb3-contrib) ────────────────

    def action_masks(self) -> np.ndarray:
        """Masque binaire sur l'espace d'action Discrete(7).

        Règles par état de jeu :

          Overworld normal  → toutes les actions autorisées.

          Transition (fade) → mouvement et Start bloqués (inputs perdus
                              pendant le fondu). Seuls 'a' et 'b' restent actifs.

          En combat         → mouvement et Start désactivés (Start n'a aucun
                              effet pendant un combat en Gen 1).
                              'a' est autorisé seulement si au moins un move
                              dispose de PP et n'est pas immunisé contre les
                              types ennemis.

        Layout ACTIONS = ['up', 'down', 'left', 'right', 'a', 'b', 'start']
                          idx 0    1      2       3       4    5      6
        """
        mask = np.ones(len(ACTIONS), dtype=bool)
        battle = self._r(RAM_BATTLE)
        fading = self._r(RAM_FADING)

        if fading:
            mask[0] = False  # up
            mask[1] = False  # down
            mask[2] = False  # left
            mask[3] = False  # right
            mask[6] = False  # start

        elif battle > 0:
            mask[0] = False  # up
            mask[1] = False  # down
            mask[2] = False  # left
            mask[3] = False  # right
            mask[5] = False  # b
            mask[6] = False  # start (sans effet en combat Gen 1)

            enemy_type1 = self._r(RAM_ENEMY_TYPE1)
            enemy_type2 = self._r(RAM_ENEMY_TYPE2)
            enemy_type_names = [
                RAM_TYPE_BYTE_TO_NAME.get(enemy_type1, "normal"),
                RAM_TYPE_BYTE_TO_NAME.get(enemy_type2, "normal"),
            ]
            has_usable_move = False
            for i in range(4):
                pp      = self._r(RAM_MOVE_PP[i])
                move_id = self._r(RAM_MOVE_IDS[i])
                if pp == 0 or move_id == 0:
                    continue
                if move_id in STATUS_MOVES:
                    continue
                move_type_byte = MOVE_TYPES.get(move_id, 0x00)
                move_type_name = RAM_TYPE_BYTE_TO_NAME.get(move_type_byte, "normal")
                mult = 1.0
                for def_type in enemy_type_names:
                    mult *= TYPE_CHART.get(move_type_name, {}).get(def_type, 1.0)
                if mult > 0.0:
                    has_usable_move = True
                    break
            if not has_usable_move:
                mask[4] = False

            if not mask.any():
                mask[4] = True

        return mask
