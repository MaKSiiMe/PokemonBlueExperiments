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
from src.agent.video_callback import VideoRecorderCallback


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
        ram_only:      bool = False,
    ):
        """
        Args:
            env_factory   : callable sans argument retournant un PokemonBlueEnv.
            model_path    : chemin vers un modèle .zip existant (optionnel).
            n_envs        : nombre d'envs parallèles (défaut 16, calibré 12GB WSL2).
            device        : 'auto' sélectionne CUDA si disponible, sinon CPU.
            backend       : VecBackend.SUBPROC (recommandé) ou VecBackend.DUMMY.
            compile_model : active torch.compile sur la politique (CUDA requis).
            ram_only      : si True, utilise une MLP sur le vecteur RAM (16,) au lieu
                            de la CNN+GRU — ~10× plus rapide, idéal pour explorer la
                            fonction de récompense rapidement.
        """
        self.vec_env  = make_vec_env(env_fns=[env_factory] * n_envs, backend=backend)
        self.ram_only = ram_only

        # Sélection de la politique selon le mode
        if ram_only:
            policy        = 'MlpPolicy'
            policy_kwargs = {'net_arch': [256, 256, 256]}
            mode_label    = 'RAM-only MLP'
        else:
            policy        = PokemonGRUPolicy
            policy_kwargs = None
            mode_label    = 'CNN+GRU'

        if model_path and os.path.exists(model_path):
            print(f"[Exploration] Loading model: {model_path}")
            custom = {} if ram_only else {'policy_class': PokemonGRUPolicy}
            self.model = MaskablePPO.load(
                model_path,
                env=self.vec_env,
                device=device,
                custom_objects=custom,
            )
        else:
            print(f"[Exploration] New MaskablePPO — {mode_label} | n_envs={n_envs} | backend={backend}")
            self.model = MaskablePPO(
                policy          = policy,
                policy_kwargs   = policy_kwargs,
                env             = self.vec_env,
                learning_rate   = 3e-4,
                n_steps         = 4096,
                batch_size      = 256,
                n_epochs        = 4,
                gamma           = 0.999,
                gae_lambda      = 0.95,
                clip_range      = 0.1,
                ent_coef        = 0.05,
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
        total_timesteps:  int = 500_000,
        save_dir:         str = 'models/rl_checkpoints/',
        save_path:        str | None = None,
        reset_timesteps:  bool = True,
        log_freq:         int = 1_000,
        monitor_verbose:  int = 1,
        env_factory_video = None,
        video_freq:       int = 200_000,
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
        callbacks = [checkpoint_cb, metrics_cb]

        if env_factory_video is not None:
            video_cb = VideoRecorderCallback(
                env_factory = env_factory_video,
                record_freq = video_freq,
                n_steps     = 500,
            )
            callbacks.append(video_cb)

        self.model.learn(
            total_timesteps     = total_timesteps,
            callback            = callbacks,
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
