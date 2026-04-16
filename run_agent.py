"""
run_agent.py — Point d'entrée principal.

Objectif : partir du labo du Prof. Chen (Pokédex obtenu) et battre Brock.

Usage :
    python run_agent.py                        # inférence
    python run_agent.py --render               # avec fenêtre SDL2
    python run_agent.py --train                # entraîne (2 phases)
    python run_agent.py --train --steps 1000000
    python run_agent.py --model models/rl_checkpoints/final.zip
"""

import argparse
import os
from functools import partial

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

ROM_PATH   = 'ROMs/PokemonBlue.gb'
# Save state : Bourg Palette, starter reçu, parcel livré, Pokédex obtenu.
INIT_STATE = 'states/00_pallet_town.state'
MAX_STEPS  = 100_000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--state',    default=None,        help='Save state à charger')
    p.add_argument('--model',    default=None,        help='Modèle PPO (.zip)')
    p.add_argument('--train',    action='store_true', help="Entraîner l'agent")
    p.add_argument('--steps',    type=int, default=500_000, help='Steps total (phases 1+2)')
    p.add_argument('--no-finetune', action='store_true', help='Sauter la phase 2')
    p.add_argument('--n-envs',   type=int, default=12,
                   help='Environnements parallèles (défaut: 12 — calibré 12GB WSL2 + FRAME_SKIP=1)')
    p.add_argument('--backend',  default='subproc',
                   choices=['dummy', 'subproc'],
                   help='Backend de vectorisation (défaut: subproc)')
    p.add_argument('--compile',  action='store_true',
                   help='Active torch.compile sur la politique (CUDA requis)')
    p.add_argument('--render',   action='store_true', help='Afficher la fenêtre SDL2')
    p.add_argument('--speed',    type=int, default=0, help='Vitesse émulateur (0=max)')
    return p.parse_args()


def make_env(
    state: str,
    headless: bool = True,
    speed: int = 0,
    max_steps: int = 2000,
    monitor: bool = True,
    kg=None,
    archive=None,
    use_archive_prob: float = 0.5,
    capture_every: int = 4,
):
    """Factory d'environnement.

    Args:
        archive          : CellArchive Go-Explore (optionnel).
        use_archive_prob : probabilité de téléportation au reset.
        capture_every    : capture le savestate tous les N steps.
    """
    from src.emulator.pokemon_env import PokemonBlueEnv
    env = PokemonBlueEnv(
        rom_path   = ROM_PATH,
        init_state = state,
        headless   = headless,
        speed      = speed,
        max_steps  = max_steps,
        kg         = kg,
    )

    if archive is not None:
        from src.agent.go_explore import GoExploreWrapper
        env = GoExploreWrapper(
            env,
            archive=archive,
            use_archive_prob=use_archive_prob,
            capture_every=capture_every,
        )

    return Monitor(env) if monitor else env


def run_train(args):
    from src.agent.exploration_agent import ExplorationAgent

    state    = args.state or INIT_STATE
    save_dir = 'models/rl_checkpoints/'
    os.makedirs(save_dir, exist_ok=True)

    # Phase 1 — exploration large (budget 60%, épisodes longs)
    max_ep_p1 = 8000
    steps_p1  = int(args.steps * 0.6)
    print(f"[Train] Objectif : battre Brock (Badge Pierre)")
    print(f"[Train] State    : {state}")
    print(f"[Train] Phase 1  : {steps_p1} steps  max/ep={max_ep_p1}  n_envs={args.n_envs}")

    factory_p1 = partial(make_env, state, max_steps=max_ep_p1)
    agent = ExplorationAgent(
        factory_p1,
        model_path    = args.model,
        n_envs        = args.n_envs,
        backend       = args.backend,
        compile_model = args.compile,
    )
    p1_path = os.path.join(save_dir, 'phase1.zip')
    agent.train(total_timesteps=steps_p1, save_path=p1_path)
    agent.close()
    del agent

    if args.no_finetune:
        return

    # Phase 2 — fine-tune (budget 40%, épisodes plus courts)
    max_ep_p2 = 2000
    steps_p2  = args.steps - steps_p1
    print(f"\n[Train] Phase 2  : {steps_p2} steps  max/ep={max_ep_p2}  n_envs={args.n_envs}")

    factory_p2 = partial(make_env, state, max_steps=max_ep_p2)
    agent2 = ExplorationAgent(
        factory_p2,
        model_path    = p1_path,
        n_envs        = args.n_envs,
        backend       = args.backend,
        compile_model = args.compile,
    )
    agent2.train(
        total_timesteps = steps_p2,
        save_path       = os.path.join(save_dir, 'final.zip'),
        reset_timesteps = False,
    )
    agent2.close()
    print(f"\n[Train] Modèle final : {save_dir}final.zip")


def run_inference(args):
    from src.agent.exploration_agent import ExplorationAgent

    state    = args.state or INIT_STATE
    headless = not args.render

    env      = make_env(state, headless=headless, speed=args.speed, max_steps=MAX_STEPS)
    base_env = env.unwrapped
    obs, _   = env.reset()

    if args.model and os.path.exists(args.model):
        print(f"[Run] Loading model: {args.model}")
        ppo_model = MaskablePPO.load(args.model, env=env)
    else:
        print("[Run] Aucun modèle — MaskablePPO aléatoire")
        ppo_model = MaskablePPO('MlpPolicy', env=env, verbose=0)

    agent = ExplorationAgent.from_model(ppo_model, env=env)

    print(f"[Run] State : {state}  |  max steps : {MAX_STEPS}")
    print("[Run] Ctrl+C pour arrêter.\n")

    from src.emulator.ram_map import RAM_MAP_ID, RAM_PLAYER_X, RAM_PLAYER_Y, RAM_BADGES

    step = 0
    try:
        while step < MAX_STEPS:
            masks  = base_env.action_masks()
            action = agent.act(obs, action_masks=masks)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if terminated or truncated:
                obs, _ = env.reset()
                if info.get('ms_badge1'):
                    print("[Run] Badge Pierre obtenu ! Mission accomplie.")
                    break

            if step % 100 == 0:
                mid = base_env._r(RAM_MAP_ID)
                x   = base_env._r(RAM_PLAYER_X)
                y   = base_env._r(RAM_PLAYER_Y)
                print(f"[Run] step={step:6d}  map=0x{mid:02X}  x={x:3d}  y={y:3d}")

    except KeyboardInterrupt:
        pass

    env.close()
    print(f"[Run] Terminé en {step} steps.")


if __name__ == '__main__':
    if not os.path.exists(ROM_PATH):
        print(f"ROM introuvable : {ROM_PATH}")
        exit(1)

    args = parse_args()

    if not os.path.exists(INIT_STATE):
        print(f"State introuvable : {INIT_STATE}")
        import sys; sys.exit(1)

    if args.train:
        run_train(args)
    else:
        run_inference(args)
