"""
video_callback.py — Enregistre un GIF de l'agent toutes les N steps.

Toutes les `record_freq` steps, lance un rollout d'inférence de `n_steps`
sur un env isolé (non vectorisé, headless) et sauvegarde les frames en GIF
dans `logs/videos/step_<N>.gif`.

Usage :
    from src.agent.video_callback import VideoRecorderCallback

    cb = VideoRecorderCallback(
        env_factory=lambda: PokemonBlueEnv(...),
        record_freq=200_000,
        n_steps=500,
    )
    model.learn(total_timesteps=5_000_000, callback=[metrics_cb, cb])
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class VideoRecorderCallback(BaseCallback):
    """Enregistre un GIF de rollout d'inférence périodiquement.

    Args:
        env_factory  : callable sans argument qui retourne un PokemonBlueEnv.
        record_freq  : fréquence en steps entre deux enregistrements.
        n_steps      : longueur du rollout capturé (en steps agent).
        video_dir    : dossier de sortie des GIFs.
        fps          : framerate du GIF.
        verbose      : 0=silencieux, 1=affiche un message à chaque enregistrement.
    """

    def __init__(
        self,
        env_factory:  Callable,
        record_freq:  int = 200_000,
        n_steps:      int = 500,
        video_dir:    str = 'logs/videos',
        fps:          int = 15,
        verbose:      int = 1,
    ):
        super().__init__(verbose=verbose)
        self.env_factory = env_factory
        self.record_freq = record_freq
        self.n_steps     = n_steps
        self.video_dir   = video_dir
        self.fps         = fps
        self._last_record = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_record < self.record_freq:
            return True
        self._last_record = self.num_timesteps
        self._record()
        return True

    def _record(self) -> None:
        try:
            import imageio
        except ImportError:
            print("[VideoCallback] imageio non installé — pip install imageio")
            return

        os.makedirs(self.video_dir, exist_ok=True)
        env = self.env_factory()
        obs, _ = env.reset()
        frames = []

        for _ in range(self.n_steps):
            # Capture l'écran RGBA (144, 160, 4) avant chaque action
            frames.append(env.pyboy.screen.ndarray[:, :, :3].copy())

            action_masks = env.action_masks() if hasattr(env, 'action_masks') else None
            action, _ = self.model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break

        env.close()

        out_path = os.path.join(self.video_dir, f'step_{self.num_timesteps:09d}.gif')
        imageio.mimsave(out_path, frames, fps=self.fps, loop=0)

        if self.verbose >= 1:
            print(f"[VideoCallback] GIF sauvegardé : {out_path} ({len(frames)} frames)")

        # W&B optionnel
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(
                    {"video": wandb.Video(np.stack(frames).transpose(0, 3, 1, 2), fps=self.fps)},
                    step=self.num_timesteps,
                )
        except ImportError:
            pass
