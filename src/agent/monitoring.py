"""
monitoring.py — Callback de monitoring des métriques de jeu pour MaskablePPO.

Métriques loguées dans TensorBoard à chaque intervalle de log :

  ── Algorithme RL (natives SB3) ──────────────────────────────────────────────
  train/entropy_loss     — entropie de la politique (incertitude des actions)
  train/value_loss       — précision de la tête Critique
  train/policy_gradient_loss

  ── Débit ────────────────────────────────────────────────────────────────────
  perf/sps               — Steps Per Second mesuré sur la fenêtre courante

  ── Navigation (moyennées sur tous les envs parallèles) ──────────────────────
  game/unique_maps_mean  — nombre moyen de cartes distinctes visitées par épisode
  game/unique_maps_max   — max sur tous les envs actifs

  ── Progression de l'équipe ───────────────────────────────────────────────────
  game/max_level_mean    — niveau max moyen de l'équipe (indicateur de grind)
  game/max_level_max     — niveau max absolu sur tous les envs
  game/n_badges_mean     — nombre moyen de badges obtenus
  game/pokedex_mean      — nombre moyen d'espèces capturées

  ── Taux de complétion des jalons ────────────────────────────────────────────
  milestones/viridian    — fraction des envs ayant visité Bourg des Eaux
  milestones/forest      — fraction des envs ayant visité la Forêt de Jade
  milestones/pewter      — fraction des envs ayant visité Argenta
  milestones/badge1      — fraction des envs ayant obtenu le Badge Pierre
  milestones/mt_moon     — fraction des envs ayant visité Mont Sélénite

Usage :
    from src.agent.monitoring import GameMetricsCallback

    cb = GameMetricsCallback(log_freq=1000, verbose=1)
    model.learn(total_timesteps=500_000, callback=cb)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# Noms des jalons narratifs (dans l'ordre chronologique du jeu)
MILESTONE_KEYS = ['ms_viridian', 'ms_forest', 'ms_pewter', 'ms_badge1', 'ms_mt_moon']
MILESTONE_LABELS = {
    'ms_viridian': 'viridian',
    'ms_forest':   'forest',
    'ms_pewter':   'pewter',
    'ms_badge1':   'badge1',
    'ms_mt_moon':  'mt_moon',
}


class GameMetricsCallback(BaseCallback):
    """Callback SB3 qui agrège les métriques de jeu et les logue dans TensorBoard.

    Lit les champs du dict `info` retourné par PokemonBlueEnv.step() à chaque
    step de collecte, les accumule sur une fenêtre glissante, et les écrit
    dans le logger SB3 (TensorBoard ou W&B selon la configuration).

    Args:
        log_freq   : intervalle de logging en nombre de steps globaux.
        window     : taille de la fenêtre glissante pour lisser les métriques
                     bruyantes (moyenne sur les `window` dernières valeurs).
        verbose    : 0=silencieux, 1=affiche les métriques dans la console.
    """

    def __init__(
        self,
        log_freq: int = 1_000,
        window:   int = 200,
        verbose:  int = 0,
    ):
        super().__init__(verbose=verbose)
        self.log_freq = log_freq
        self.window   = window

        # Fenêtres glissantes pour toutes les métriques scalaires
        self._windows: dict[str, deque] = {
            'unique_maps':   deque(maxlen=window),
            'max_level':     deque(maxlen=window),
            'n_badges':      deque(maxlen=window),
            'pokedex_owned': deque(maxlen=window),
            **{k: deque(maxlen=window) for k in MILESTONE_KEYS},
        }

        # SPS (Steps Per Second)
        self._t_last:     float = 0.0
        self._steps_last: int   = 0

    # ── BaseCallback API ──────────────────────────────────────────────────────

    def _on_training_start(self) -> None:
        self._t_last     = time.perf_counter()
        self._steps_last = self.num_timesteps

    def _on_step(self) -> bool:
        """Appelé après chaque step de collect_rollouts.

        `self.locals['infos']` est une liste de dicts, un par env parallèle.
        """
        infos = self.locals.get('infos', [])
        if not infos:
            return True

        # Accumule les métriques de chaque env actif
        for info in infos:
            if not isinstance(info, dict):
                continue
            for key in self._windows:
                val = info.get(key)
                if val is not None:
                    self._windows[key].append(float(val))

        # Log à la fréquence définie
        if self.num_timesteps % self.log_freq == 0:
            self._log_metrics()

        return True

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_metrics(self) -> None:
        """Calcule les statistiques et les écrit dans le logger SB3."""
        now   = time.perf_counter()
        steps = self.num_timesteps

        # ── SPS ──────────────────────────────────────────────────────────────
        dt = now - self._t_last
        if dt > 0:
            sps = (steps - self._steps_last) / dt
            self.logger.record('perf/sps', sps)
        self._t_last     = now
        self._steps_last = steps

        # ── Navigation ───────────────────────────────────────────────────────
        self._record_window('game/unique_maps_mean', 'unique_maps', 'mean')
        self._record_window('game/unique_maps_max',  'unique_maps', 'max')

        # ── Progression de l'équipe ───────────────────────────────────────────
        self._record_window('game/max_level_mean',  'max_level',     'mean')
        self._record_window('game/max_level_max',   'max_level',     'max')
        self._record_window('game/n_badges_mean',   'n_badges',      'mean')
        self._record_window('game/pokedex_mean',    'pokedex_owned', 'mean')

        # ── Milestones ────────────────────────────────────────────────────────
        # Taux = fraction des envs qui ont atteint ce jalon (fenêtre glissante)
        for key, label in MILESTONE_LABELS.items():
            self._record_window(f'milestones/{label}', key, 'mean')

        self.logger.dump(step=steps)

        if self.verbose >= 1:
            maps   = self._window_stat('unique_maps', 'mean')
            levels = self._window_stat('max_level', 'max')
            badges = self._window_stat('n_badges', 'mean')
            ms_b1  = self._window_stat('ms_badge1', 'mean')
            print(
                f"[Monitor] step={steps:>9,} | "
                f"maps={maps:.1f} | "
                f"max_lvl={levels:.0f} | "
                f"badges={badges:.2f} | "
                f"badge1_rate={ms_b1:.2%}"
            )

    def _record_window(self, tag: str, key: str, stat: str) -> None:
        val = self._window_stat(key, stat)
        if val is not None:
            self.logger.record(tag, val)

    def _window_stat(self, key: str, stat: str) -> Optional[float]:
        buf = self._windows.get(key)
        if not buf:
            return None
        arr = np.array(buf)
        if stat == 'mean':
            return float(np.mean(arr))
        if stat == 'max':
            return float(np.max(arr))
        if stat == 'min':
            return float(np.min(arr))
        return None
