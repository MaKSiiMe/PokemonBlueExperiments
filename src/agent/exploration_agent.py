"""
Exploration Agent — PPO-based navigation agent.

Uses Stable Baselines3 PPO on top of PokemonBlueEnv.
Implements a waypoint curriculum: trains one objective at a time.

Waypoint format: (map_id, target_x, target_y, description)
The env reward is shaped around reaching the current waypoint.

Usage:
    from src.emulator.pokemon_env import PokemonBlueEnv
    from src.agent.exploration_agent import ExplorationAgent

    def make_env():
        return PokemonBlueEnv('ROMs/PokemonBlue.gb', init_state='states/06_pallet_town.state')

    agent = ExplorationAgent(make_env)
    agent.train(total_timesteps=500_000)
"""

from __future__ import annotations
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# ─── Waypoint curriculum ──────────────────────────────────────────────────────
# (map_id, target_x, target_y, label)
# Source: docs/roadmap.md — Bourg Palette → Badge Pierre
WAYPOINTS: list[tuple[int, int, int, str]] = [
    (0x26,  6,  2, "Chambre → escalier"),
    (0x25,  3,  7, "Maison 1F → porte sud"),
    (0x00, 10, 11, "Entrer au Labo Chen"),
    (0x52,  7,  6, "Prendre la Pokéball (starter)"),
    (0x00, 10,  6, "Sortir du Labo"),
    (0x12, 10,  5, "Route 1 — vers nord"),
    (0x01,  5, 10, "Arriver à Jadielle City"),
    (0x01,  7,  3, "Entrer au Poké Mart (colis)"),
    (0x00, 10, 11, "Retour Labo Chen (livrer colis)"),
    (0x13,  5,  5, "Route 2"),
    (0x33,  4, 12, "Forêt de Jade"),
    (0x02, 12,  8, "Argenta City"),
    (0x54,  5,  5, "Arène Argenta → Brock"),
]


class ExplorationAgent:
    """
    Wraps a PPO model with a waypoint curriculum system.

    For training  : use ExplorationAgent(env_factory, ...)
    For inference : use ExplorationAgent.from_model(ppo_model, waypoints)
    """

    def __init__(
        self,
        env_factory,
        model_path: str | None = None,
        n_envs: int = 1,
        device: str = 'cpu',
    ):
        self._waypoint_idx = 0
        self._env_factory  = env_factory

        vec_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
        self.vec_env = vec_cls([env_factory] * n_envs)

        if model_path and os.path.exists(model_path):
            print(f"[Exploration] Loading model: {model_path}")
            self.model = PPO.load(model_path, env=self.vec_env, device=device)
        else:
            print("[Exploration] Creating new PPO model")
            self.model = PPO(
                policy          = 'MlpPolicy',
                env             = self.vec_env,
                learning_rate   = 3e-4,
                n_steps         = 2048,
                batch_size      = 64,
                n_epochs        = 10,
                gamma           = 0.99,
                gae_lambda      = 0.95,
                clip_range      = 0.2,
                ent_coef        = 0.01,
                verbose         = 1,
                device          = device,
                tensorboard_log = "./logs/exploration/",
            )

    @classmethod
    def from_model(cls, ppo_model: PPO, waypoints: list) -> 'ExplorationAgent':
        """
        Create an inference-only agent from an already-loaded PPO model.
        No VecEnv is created — avoids spinning up extra PyBoy instances.
        """
        agent = object.__new__(cls)
        agent.model         = ppo_model
        agent.vec_env       = None
        agent._waypoint_idx = 0
        agent._env_factory  = None
        agent._waypoints    = waypoints
        return agent

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        total_timesteps: int = 500_000,
        save_dir: str = "models/rl_checkpoints/",
        save_path: str | None = None,
        reset_timesteps: bool = True,
        waypoint_idx: int | None = None,
    ):
        os.makedirs(save_dir, exist_ok=True)
        idx   = waypoint_idx if waypoint_idx is not None else self._waypoint_idx
        wp    = self.current_waypoint()
        label = wp[3] if wp else "?"
        print(f"[Exploration] Training waypoint {idx}: {label}")

        cb = CheckpointCallback(
            save_freq   = 50_000,
            save_path   = save_dir,
            name_prefix = f"explore_wp{idx}",
        )

        self.model.learn(
            total_timesteps     = total_timesteps,
            callback            = cb,
            reset_num_timesteps = reset_timesteps,
        )

        path = save_path or os.path.join(save_dir, f"wp{idx}_final.zip")
        self.model.save(path)
        print(f"[Exploration] Saved → {path}")
        return path

    # ── Inference ─────────────────────────────────────────────────────────────

    def act(self, obs: np.ndarray) -> int:
        """Return action index (deterministic, for inference/eval)."""
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    # ── Curriculum ────────────────────────────────────────────────────────────

    @property
    def _wps(self) -> list:
        """Waypoint list — supports both training mode (WAYPOINTS) and inference mode."""
        return getattr(self, '_waypoints', WAYPOINTS)

    def current_waypoint(self) -> tuple | None:
        if self._waypoint_idx < len(self._wps):
            return self._wps[self._waypoint_idx]
        return None

    def advance_waypoint(self):
        """Call when the agent successfully reaches the current waypoint."""
        if self._waypoint_idx < len(self._wps) - 1:
            self._waypoint_idx += 1
            wp = self._wps[self._waypoint_idx]
            print(f"[Exploration] Waypoint reached → next: [{self._waypoint_idx}] {wp[3]}")
        else:
            print("[Exploration] All waypoints complete!")

    def waypoint_reached(self, map_id: int, x: int, y: int, tol: int = 3) -> bool:
        """Check if the player is within `tol` tiles of the current waypoint."""
        wp = self.current_waypoint()
        if wp is None:
            return False
        wp_map, wp_x, wp_y = wp[0], wp[1], wp[2]
        return map_id == wp_map and abs(x - wp_x) <= tol and abs(y - wp_y) <= tol
