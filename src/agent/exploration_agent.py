"""
Exploration Agent — Agent de navigation PPO (MaskablePPO, sb3-contrib).

Objectif : partir du labo du Prof. Chen (après Pokédex) et battre Brock.
Entraînement en deux phases :
  Phase 1 — exploration large (max_steps élevé, budget 60%)
  Phase 2 — fine-tune      (max_steps réduit, budget 40%)

Usage :
    agent = ExplorationAgent(env_factory)
    agent.train(total_timesteps=500_000)
"""

from __future__ import annotations
import os
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback

from src.agent.custom_policy import PokemonGRUPolicy
from src.agent.vectorization import VecBackend, make_vec_env
from src.agent.monitoring import GameMetricsCallback


class ExplorationAgent:
    """Wraps MaskablePPO pour l'exploration Pokémon Bleu."""

    def __init__(
        self,
        env_factory,
        model_path:    str | None = None,
        n_envs:        int = 16,
        device:        str = 'auto',
        backend:       VecBackend | str = VecBackend.SUBPROC,
        compile_model: bool = False,
    ):
        """
        Args:
            env_factory   : callable sans argument retournant un PokemonBlueEnv.
            model_path    : chemin vers un modèle .zip existant (optionnel).
            n_envs        : nombre d'envs parallèles (défaut 16, calibré 12GB WSL2).
            device        : 'auto' sélectionne CUDA si disponible, sinon CPU.
            backend       : VecBackend.SUBPROC (recommandé) ou VecBackend.DUMMY.
            compile_model : active torch.compile sur la politique (CUDA requis).
        """
        self.vec_env = make_vec_env(
            env_fns=[env_factory] * n_envs,
            backend=backend,
        )

        if model_path and os.path.exists(model_path):
            print(f"[Exploration] Loading model: {model_path}")
            self.model = MaskablePPO.load(
                model_path,
                env=self.vec_env,
                device=device,
                custom_objects={'policy_class': PokemonGRUPolicy},
            )
        else:
            print(f"[Exploration] New MaskablePPO — CNN+GRU | n_envs={n_envs} | backend={backend}")
            self.model = MaskablePPO(
                policy          = PokemonGRUPolicy,
                env             = self.vec_env,
                learning_rate   = 3e-4,
                n_steps         = 2048,
                batch_size      = 64,
                n_epochs        = 3,
                gamma           = 0.997,
                gae_lambda      = 0.95,
                clip_range      = 0.2,
                ent_coef        = 0.02,
                verbose         = 1,
                device          = device,
                tensorboard_log = './logs/exploration/',
            )

        if compile_model:
            try:
                import torch
                if torch.cuda.is_available():
                    self.model.policy = torch.compile(self.model.policy)
                    print("[Exploration] torch.compile activé sur la politique.")
                else:
                    print("[Exploration] torch.compile ignoré (CUDA non disponible).")
            except Exception as exc:
                print(f"[Exploration] torch.compile échoué (ignoré) : {exc}")

    @classmethod
    def from_model(cls, ppo_model, env=None) -> 'ExplorationAgent':
        """Crée un agent inférence-only depuis un modèle déjà chargé."""
        agent = object.__new__(cls)
        agent.model   = ppo_model
        agent.vec_env = env
        return agent

    # ── Entraînement ──────────────────────────────────────────────────────────

    def train(
        self,
        total_timesteps: int = 500_000,
        save_dir:        str = 'models/rl_checkpoints/',
        save_path:       str | None = None,
        reset_timesteps: bool = True,
        log_freq:        int = 1_000,
        monitor_verbose: int = 1,
    ) -> str:
        """Lance l'entraînement PPO avec checkpoint et monitoring.

        Returns:
            Chemin du modèle sauvegardé.
        """
        os.makedirs(save_dir, exist_ok=True)

        checkpoint_cb = CheckpointCallback(
            save_freq   = 50_000,
            save_path   = save_dir,
            name_prefix = 'explore',
        )
        metrics_cb = GameMetricsCallback(
            log_freq = log_freq,
            verbose  = monitor_verbose,
        )

        self.model.learn(
            total_timesteps     = total_timesteps,
            callback            = [checkpoint_cb, metrics_cb],
            reset_num_timesteps = reset_timesteps,
        )

        path = save_path or os.path.join(save_dir, 'final.zip')
        self.model.save(path)
        print(f"[Exploration] Saved → {path}")
        return path

    def close(self) -> None:
        if hasattr(self, 'vec_env') and self.vec_env is not None:
            self.vec_env.close()

    # ── Inférence ─────────────────────────────────────────────────────────────

    def act(self, obs: np.ndarray, action_masks: np.ndarray | None = None) -> int:
        action, _ = self.model.predict(obs, deterministic=True, action_masks=action_masks)
        return int(action)
