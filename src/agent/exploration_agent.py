"""
Exploration Agent — Agent de navigation PPO (Stable Baselines3).

Implémente un curriculum de waypoints : entraîne un objectif à la fois.
La liste de waypoints est importée depuis src.agent.waypoints (source unique).

Usage :
    agent = ExplorationAgent(env_factory)
    agent.train(total_timesteps=500_000)
"""

from __future__ import annotations
import os
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from src.agent.waypoints import WAYPOINTS


class ExplorationAgent:
    """
    Wraps PPO avec un curriculum de waypoints.

    Deux modes d'instanciation :
      - Entraînement : ExplorationAgent(env_factory, ...)
      - Inférence    : ExplorationAgent.from_model(ppo_model, waypoints)
    """

    def __init__(
        self,
        env_factory,
        model_path: str | None = None,
        n_envs: int = 1,
        device: str = 'cpu',
    ):
        self._waypoint_idx = 0

        vec_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
        self.vec_env = vec_cls([env_factory] * n_envs)

        if model_path and os.path.exists(model_path):
            print(f"[Exploration] Loading model: {model_path}")
            self.model = MaskablePPO.load(model_path, env=self.vec_env, device=device)
        else:
            print("[Exploration] Creating new MaskablePPO model")
            self.model = MaskablePPO(
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
                tensorboard_log = './logs/exploration/',
            )

    @classmethod
    def from_model(cls, ppo_model: PPO, waypoints: list) -> 'ExplorationAgent':
        """Crée un agent inférence-only depuis un modèle déjà chargé (pas de VecEnv)."""
        agent = object.__new__(cls)
        agent.model         = ppo_model
        agent.vec_env       = None
        agent._waypoint_idx = 0
        agent._waypoints    = waypoints
        return agent

    # ── Entraînement ──────────────────────────────────────────────────────────

    def train(
        self,
        total_timesteps: int = 500_000,
        save_dir: str = 'models/rl_checkpoints/',
        save_path: str | None = None,
        reset_timesteps: bool = True,
        waypoint_idx: int | None = None,
    ):
        os.makedirs(save_dir, exist_ok=True)
        idx   = waypoint_idx if waypoint_idx is not None else self._waypoint_idx
        wp    = self.current_waypoint()
        label = wp[4] if wp else '?'   # index 4 = label dans le format (map,x,y,state,label,steps)
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

    def close(self) -> None:
        if hasattr(self, 'model') and self.model.env is not None:
            self.model.env.close()

    # ── Inférence ─────────────────────────────────────────────────────────────

    def act(self, obs: np.ndarray, action_masks: np.ndarray | None = None) -> int:
        action, _ = self.model.predict(obs, deterministic=True, action_masks=action_masks)
        return int(action)

    # ── Curriculum ────────────────────────────────────────────────────────────

    @property
    def _wps(self) -> list:
        return getattr(self, '_waypoints', WAYPOINTS)

    def current_waypoint(self) -> tuple | None:
        if self._waypoint_idx < len(self._wps):
            return self._wps[self._waypoint_idx]
        return None

    def advance_waypoint(self):
        if self._waypoint_idx < len(self._wps) - 1:
            self._waypoint_idx += 1
            wp = self._wps[self._waypoint_idx]
            print(f"[Exploration] Waypoint atteint → [{self._waypoint_idx}] {wp[4]}")
        else:
            print("[Exploration] Curriculum terminé !")

    def waypoint_reached(self, map_id: int, x: int, y: int, tol: int = 1) -> bool:
        wp = self.current_waypoint()
        if wp is None:
            return False
        return map_id == wp[0] and abs(x - wp[1]) <= tol and abs(y - wp[2]) <= tol
