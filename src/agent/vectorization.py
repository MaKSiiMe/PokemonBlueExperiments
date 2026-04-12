"""
vectorization.py — Couche d'abstraction pour la vectorisation des environnements.

Deux backends disponibles :

  subproc   (défaut)
    SB3's SubprocVecEnv — un processus Python par env, communication via
    multiprocessing.Pipe. Simple et robuste, mais la sérialisation pickle
    et la synchronisation inter-processus limitent le débit à ~5 000 SPS
    sur 12 envs.

  pufferlib  (recommandé pour la production)
    PufferLib — vectorisation en mémoire partagée avec un backend C optimisé.
    Les observations de N envs PyBoy sont fusionnées directement dans des
    tenseurs préalloués sans sérialisation pickle. La surcharge IPC est
    quasi-nulle. Permet d'atteindre 10 000–30 000 SPS selon le matériel.
    Nécessite : pip install pufferlib>=0.9

Usage :
    from src.agent.vectorization import make_vec_env, VecBackend

    # Backend SB3 (fallback)
    vec = make_vec_env(env_fns, backend=VecBackend.SUBPROC)

    # Backend PufferLib (production)
    vec = make_vec_env(env_fns, backend=VecBackend.PUFFERLIB)
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, List

import gymnasium as gym


class VecBackend(str, Enum):
    """Backend de vectorisation des environnements."""
    DUMMY      = 'dummy'       # un seul processus (debug)
    SUBPROC    = 'subproc'     # SubprocVecEnv SB3
    PUFFERLIB  = 'pufferlib'   # PufferLib mémoire partagée (production)


def make_vec_env(
    env_fns: List[Callable[[], gym.Env]],
    backend: VecBackend | str = VecBackend.SUBPROC,
) -> gym.Env:
    """Crée un environnement vectorisé avec le backend spécifié.

    Args:
        env_fns : liste de callables sans arguments, chacun retournant un
                  Gymnasium Env. La longueur détermine le nombre d'envs
                  parallèles (num_envs).
        backend : backend de vectorisation (VecBackend ou str).

    Returns:
        VecEnv compatible SB3 (pour DUMMY/SUBPROC) ou wrapper PufferLib.

    Raises:
        ImportError  : si pufferlib n'est pas installé et backend=PUFFERLIB.
        ValueError   : si backend inconnu.
    """
    backend = VecBackend(backend)

    if backend == VecBackend.DUMMY:
        from stable_baselines3.common.vec_env import DummyVecEnv
        return DummyVecEnv(env_fns)

    if backend == VecBackend.SUBPROC:
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        if len(env_fns) == 1:
            return DummyVecEnv(env_fns)
        return SubprocVecEnv(env_fns)

    if backend == VecBackend.PUFFERLIB:
        return _make_pufferlib_vec_env(env_fns)

    raise ValueError(f"Backend inconnu : {backend}")


def _make_pufferlib_vec_env(env_fns: List[Callable[[], gym.Env]]) -> gym.Env:
    """Crée un VecEnv PufferLib compatible SB3.

    PufferLib utilise la mémoire partagée pour transférer les observations
    des N workers PyBoy directement dans des tenseurs préalloués, sans
    sérialisation pickle — d'où le gain de débit massif.

    Le wrapper `SB3VecEnvAdapter` rend le VecEnv PufferLib compatible avec
    l'interface SB3 (reset_infos, env_method, etc.) attendue par MaskablePPO.

    Args:
        env_fns : callables retournant des Gymnasium Env.

    Returns:
        VecEnv compatible SB3.
    """
    try:
        import pufferlib.vector as puf_vec
    except ImportError as exc:
        raise ImportError(
            "PufferLib n'est pas installé. Installez-le avec :\n"
            "  pip install pufferlib>=0.9\n"
            "ou ajoutez 'pufferlib>=0.9' aux dépendances du projet."
        ) from exc

    # PufferLib attend une factory callable (pas une liste de callables)
    env_creator = env_fns[0]
    num_envs    = len(env_fns)

    puf_envs = puf_vec.make(
        env_creator,
        backend=puf_vec.Multiprocessing,
        num_envs=num_envs,
    )

    return SB3VecEnvAdapter(puf_envs)


class SB3VecEnvAdapter:
    """Adapte un VecEnv PufferLib à l'interface SB3.

    SB3's MaskablePPO appelle des méthodes spécifiques (env_method,
    get_attr, reset_infos) que PufferLib n'expose pas. Ce wrapper
    fournit ces méthodes en les déléguant ou en les simulant.

    Note : pour tirer pleinement parti de PufferLib, migrez vers son
    propre training loop (cf. pipeline.md §7.1 et puffer_trainer.py).
    """

    def __init__(self, puf_envs):
        self._envs       = puf_envs
        self.num_envs    = puf_envs.num_envs
        # Expose les espaces depuis le premier env sous-jacent
        dummy = puf_envs.driver_env
        self.observation_space = dummy.observation_space
        self.action_space      = dummy.action_space

    # ── Interface minimale SB3 VecEnv ────────────────────────────────────────

    def reset(self):
        obs, infos = self._envs.reset()
        return obs

    def step(self, actions):
        self._envs.send(actions)
        obs, rewards, terminated, truncated, infos = self._envs.recv()
        dones = terminated | truncated
        return obs, rewards, dones, infos

    def close(self):
        self._envs.close()

    def env_method(self, method_name, *args, indices=None, **kwargs):
        """Délègue les appels de méthodes aux envs sous-jacents."""
        results = []
        envs = self._envs.envs if indices is None else [self._envs.envs[i] for i in indices]
        for env in envs:
            results.append(getattr(env, method_name)(*args, **kwargs))
        return results

    def get_attr(self, attr_name, indices=None):
        envs = self._envs.envs if indices is None else [self._envs.envs[i] for i in indices]
        return [getattr(env, attr_name) for env in envs]

    def set_attr(self, attr_name, value, indices=None):
        envs = self._envs.envs if indices is None else [self._envs.envs[i] for i in indices]
        for env in envs:
            setattr(env, attr_name, value)

    def seed(self, seed=None):
        pass   # PyBoy n'utilise pas de seed RNG

    @property
    def reset_infos(self):
        return [{}] * self.num_envs

    def __len__(self):
        return self.num_envs
