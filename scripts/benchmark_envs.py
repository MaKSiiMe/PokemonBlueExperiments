"""
benchmark_envs.py — Calibration du nombre optimal d'envs parallèles.

Compare SubprocVecEnv (SB3) vs PufferLib pour trouver la configuration
maximisant le SPS (Steps Per Second) sur le matériel courant.

Utilise PokemonBlueEnv si la ROM est disponible, sinon un env synthétique
(même espace d'observation) comme proxy de timing.

Usage :
    python scripts/benchmark_envs.py
    python scripts/benchmark_envs.py --steps 200 --rom ROMs/PokemonBlue.gb
    python scripts/benchmark_envs.py --backend subproc      # SB3 seulement
    python scripts/benchmark_envs.py --backend pufferlib    # PufferLib seulement
    python scripts/benchmark_envs.py --backend both         # comparaison complète
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

ROM_PATH   = 'ROMs/PokemonBlue.gb'
INIT_STATE = 'states/00_pallet_town.state'

# Candidats testés pour SubprocVecEnv
CANDIDATES_SUBPROC   = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24]
# PufferLib supporte plus d'envs sans overhead IPC → on monte plus haut
CANDIDATES_PUFFERLIB = [4, 8, 12, 16, 20, 24, 32, 48, 64]

STEPS_PER_TRIAL = 100   # steps par env pour mesurer le débit


# ── Env factories ─────────────────────────────────────────────────────────────

def make_pokemon_env():
    """Factory pour PokemonBlueEnv (nécessite la ROM)."""
    from src.emulator.pokemon_env import PokemonBlueEnv
    return PokemonBlueEnv(
        rom_path   = ROM_PATH,
        init_state = INIT_STATE,
        headless   = True,
        speed      = 0,
        max_steps  = STEPS_PER_TRIAL + 10,
    )


def make_synthetic_env():
    """Factory pour un env synthétique avec le même espace d'observation.

    Utilisé quand la ROM est absente — simule le coût de sérialisation/
    désérialisation des observations sans l'émulation PyBoy.
    """
    import gymnasium as gym
    from gymnasium import spaces

    class SyntheticPokemonEnv(gym.Env):
        """Simule l'espace Dict de PokemonBlueEnv avec des observations aléatoires."""

        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Dict({
                'screen':       spaces.Box(0, 1, (3, 72, 80),  dtype=np.float32),
                'visited_mask': spaces.Box(0, 1, (1, 48, 48),  dtype=np.float32),
                'ram':          spaces.Box(0, 1, (16,),         dtype=np.float32),
            })
            self.action_space = spaces.Discrete(7)
            self._step = 0

        def reset(self, seed=None, options=None):
            self._step = 0
            return self._obs(), {}

        def step(self, action):
            self._step += 1
            done = self._step >= STEPS_PER_TRIAL
            _ = np.random.rand(72 * 80).sum()
            return self._obs(), 0.0, done, False, {}

        def _obs(self):
            return {
                'screen':       np.random.rand(3, 72, 80).astype(np.float32),
                'visited_mask': np.random.rand(1, 48, 48).astype(np.float32),
                'ram':          np.random.rand(16).astype(np.float32),
            }

    return SyntheticPokemonEnv()


# ── Benchmark SubprocVecEnv (SB3) ─────────────────────────────────────────────

def benchmark_subproc(n_envs: int, env_fn, steps: int) -> float:
    """Mesure le SPS pour n_envs envs parallèles avec SubprocVecEnv.

    Returns:
        SPS moyen (steps par seconde) sur l'ensemble des envs, ou 0.0 si erreur.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    vec_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    try:
        vec = vec_cls([env_fn] * n_envs)
        vec.reset()

        for _ in range(5):
            actions = np.array([vec.action_space.sample() for _ in range(n_envs)])
            vec.step(actions)

        t0 = time.perf_counter()
        total_steps = 0
        vec.reset()
        for _ in range(steps):
            actions = np.array([vec.action_space.sample() for _ in range(n_envs)])
            vec.step(actions)
            total_steps += n_envs
        dt = time.perf_counter() - t0

        vec.close()
        return total_steps / dt

    except Exception as e:
        print(f"    Erreur subproc n_envs={n_envs}: {e}")
        return 0.0


# ── Benchmark PufferLib ───────────────────────────────────────────────────────

def benchmark_pufferlib(n_envs: int, env_fn, steps: int) -> float:
    """Mesure le SPS avec le backend PufferLib (mémoire partagée C).

    PufferLib évite la sérialisation pickle des observations en utilisant
    des buffers de mémoire partagée. Le gain est surtout visible sur des
    observations volumineuses (screen 72×80×3 ici).

    Returns:
        SPS moyen, ou 0.0 si PufferLib n'est pas installé ou erreur.
    """
    try:
        import pufferlib.vector as puf_vec
    except ImportError:
        return 0.0

    try:
        # Crée un env de référence pour récupérer l'action_space
        ref_env = env_fn()
        action_space = ref_env.action_space
        ref_env.close()

        # PufferLib v3: GymnasiumPufferEnv est le creator, env_fn passé via env_kwargs
        # make() appelle creator(*env_args, buf=buf, seed=seed, **env_kwargs)
        # overwork=True : autorise plus de workers que de cœurs physiques
        envs = puf_vec.make(
            puf_vec.GymnasiumPufferEnv,
            env_kwargs={'env_creator': env_fn},
            num_envs=n_envs,
            backend=puf_vec.Multiprocessing,
            overwork=True,
        )
        obs, _ = envs.reset()

        # Warmup
        for _ in range(5):
            actions = np.array([action_space.sample() for _ in range(n_envs)])
            envs.step(actions)

        # Mesure
        t0 = time.perf_counter()
        total_steps = 0
        envs.reset()
        for _ in range(steps):
            actions = np.array([action_space.sample() for _ in range(n_envs)])
            envs.step(actions)
            total_steps += n_envs
        dt = time.perf_counter() - t0

        envs.close()
        return total_steps / dt

    except Exception as e:
        print(f"    Erreur pufferlib n_envs={n_envs}: {e}")
        return 0.0


# ── Analyse et affichage ───────────────────────────────────────────────────────

def print_results_table(
    subproc_results:    list[tuple[int, float]],
    pufferlib_results:  list[tuple[int, float]],
):
    """Affiche un tableau comparatif SubprocVecEnv vs PufferLib."""

    # Rassemble tous les n_envs uniques dans l'ordre
    all_n = sorted(set(n for n, _ in subproc_results + pufferlib_results))
    sub_map = dict(subproc_results)
    puf_map = dict(pufferlib_results)

    best_sub = max((s for s in sub_map.values() if s > 0), default=0)
    best_puf = max((s for s in puf_map.values() if s > 0), default=0)
    overall_best = max(best_sub, best_puf)

    print()
    print("─" * 68)
    print(f"  {'n_envs':>7}  {'SubprocVecEnv':>14}  {'PufferLib':>14}  {'Best':>6}")
    print(f"  {'─'*7}  {'─'*14}  {'─'*14}  {'─'*6}")

    for n in all_n:
        sub_s = sub_map.get(n, 0.0)
        puf_s = puf_map.get(n, 0.0)
        best  = max(sub_s, puf_s)
        marker = ' ◄' if best == overall_best and best > 0 else ''

        sub_str = f"{sub_s:>12,.0f}" if sub_s > 0 else f"{'—':>12}"
        puf_str = f"{puf_s:>12,.0f}" if puf_s > 0 else f"{'—':>12}"
        best_tag = 'sub' if sub_s >= puf_s and sub_s > 0 else 'puf' if puf_s > 0 else '—'

        print(f"  {n:>7}  {sub_str}  {puf_str}  {best_tag:>4}{marker}")

    print("─" * 68)

    # Recommandations
    best_sub_n = max(sub_map, key=sub_map.get) if sub_map else None
    best_puf_n = max(puf_map, key=puf_map.get) if puf_map else None

    print()
    print("  Résultats :")
    if best_sub_n and best_sub > 0:
        print(f"    SubprocVecEnv optimal : n_envs={best_sub_n}  ({best_sub:,.0f} SPS)")
    if best_puf_n and best_puf > 0:
        print(f"    PufferLib optimal     : n_envs={best_puf_n}  ({best_puf:,.0f} SPS)")

    if overall_best > 0:
        print()
        if best_puf > best_sub:
            gain = (best_puf - best_sub) / best_sub * 100 if best_sub > 0 else float('inf')
            print(f"  Gagnant : PufferLib +{gain:.0f}% vs SubprocVecEnv")
            print(f"    → Utilisez : --n-envs {best_puf_n} --backend pufferlib")
        elif best_sub > best_puf:
            print(f"  Gagnant : SubprocVecEnv")
            print(f"    → Utilisez : --n-envs {best_sub_n} --backend subproc")
        else:
            print(f"  Égalité SubprocVecEnv / PufferLib")

    print()
    print("  Commande d'entraînement recommandée :")
    if best_puf > best_sub and best_puf_n:
        print(f"    python run_agent.py --train --n-envs {best_puf_n} --backend pufferlib")
    elif best_sub_n:
        print(f"    python run_agent.py --train --n-envs {best_sub_n} --backend subproc")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark(args):
    n_cpu_logical  = os.cpu_count() or 12
    n_cpu_physical = n_cpu_logical // 2

    print("=" * 68)
    print("  Calibration n_envs — PokemonBlue RL")
    print("=" * 68)
    print(f"  CPU : {n_cpu_logical} threads logiques / {n_cpu_physical} cœurs physiques")
    print(f"  Steps par trial : {args.steps}")
    print(f"  Backend(s) : {args.backend}")
    print()

    rom_available = os.path.exists(args.rom)
    if rom_available:
        print(f"  ROM trouvée : {args.rom} → benchmark réel PyBoy")
        env_fn = make_pokemon_env
        mode   = 'PyBoy'
    else:
        print(f"  ROM absente → benchmark synthétique (proxy de timing IPC)")
        env_fn = make_synthetic_env
        mode   = 'Synthétique'
    print()

    do_subproc    = args.backend in ('subproc', 'both')
    do_pufferlib  = args.backend in ('pufferlib', 'both')

    subproc_results:   list[tuple[int, float]] = []
    pufferlib_results: list[tuple[int, float]] = []

    # ── SubprocVecEnv ──────────────────────────────────────────────────────────
    if do_subproc:
        print("  [SubprocVecEnv]")
        candidates = [c for c in CANDIDATES_SUBPROC if c <= n_cpu_logical + 4]
        for n in candidates:
            print(f"    n_envs={n:>3} ... ", end='', flush=True)
            sps = benchmark_subproc(n, env_fn, args.steps)
            subproc_results.append((n, sps))
            print(f"{sps:>9,.0f} SPS")
        print()

    # ── PufferLib ──────────────────────────────────────────────────────────────
    if do_pufferlib:
        try:
            import pufferlib
            print(f"  [PufferLib v{pufferlib.__version__}]")
        except ImportError:
            print("  [PufferLib] non installé — pip install pufferlib")
            do_pufferlib = False

    if do_pufferlib:
        candidates = CANDIDATES_PUFFERLIB
        for n in candidates:
            print(f"    n_envs={n:>3} ... ", end='', flush=True)
            sps = benchmark_pufferlib(n, env_fn, args.steps)
            pufferlib_results.append((n, sps))
            print(f"{sps:>9,.0f} SPS")
        print()

    # ── Tableau comparatif ────────────────────────────────────────────────────
    if subproc_results or pufferlib_results:
        print_results_table(subproc_results, pufferlib_results)

        # Résumé pour un seul backend
        if subproc_results and not pufferlib_results:
            best_n, best_sps = max(subproc_results, key=lambda r: r[1])
            efficient = [(n, s) for n, s in subproc_results if s >= 0.80 * best_sps]
            if efficient:
                lo = min(n for n, _ in efficient)
                hi = max(n for n, _ in efficient)
                print(f"  Zone 80% efficace : n_envs {lo}–{hi}")
        elif pufferlib_results and not subproc_results:
            best_n, best_sps = max(pufferlib_results, key=lambda r: r[1])
            efficient = [(n, s) for n, s in pufferlib_results if s >= 0.80 * best_sps]
            if efficient:
                lo = min(n for n, _ in efficient)
                hi = max(n for n, _ in efficient)
                print(f"  Zone 80% efficace : n_envs {lo}–{hi}")


def main():
    p = argparse.ArgumentParser(description='Calibration n_envs optimal')
    p.add_argument('--steps',   type=int, default=STEPS_PER_TRIAL,
                   help=f'Steps de mesure par trial (défaut: {STEPS_PER_TRIAL})')
    p.add_argument('--rom',     default=ROM_PATH,
                   help=f'Chemin vers la ROM (défaut: {ROM_PATH})')
    p.add_argument('--backend', default='both',
                   choices=['subproc', 'pufferlib', 'both'],
                   help='Backend(s) à tester (défaut: both)')
    args = p.parse_args()
    run_benchmark(args)


if __name__ == '__main__':
    main()
