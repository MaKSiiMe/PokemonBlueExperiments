"""
run_agent.py — Point d'entrée principal.

Usage :
    python run_agent.py                          # inférence depuis Bourg Palette
    python run_agent.py --render                 # avec fenêtre SDL2
    python run_agent.py --train                  # entraîne waypoint 0
    python run_agent.py --train --waypoint 1     # entraîne waypoint 1
    python run_agent.py --train --chain 0 1 2    # entraîne WP0+WP1+WP2 en chaîne
    python run_agent.py --train --steps 500000 --waypoint 0
    python run_agent.py --model models/rl_checkpoints/wp0_final.zip
"""

import argparse
import os
from functools import partial

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from src.agent.waypoints import WAYPOINTS

ROM_PATH   = 'ROMs/PokemonBlue.gb'
INIT_STATE = 'states/00_pallet_town.state'
MAX_STEPS  = 100_000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--state',       default=None,        help='Save state à charger')
    p.add_argument('--model',       default=None,        help='Modèle PPO (.zip)')
    p.add_argument('--train',       action='store_true', help="Entraîner l'agent")
    p.add_argument('--waypoint',    type=int, default=0, help='Index waypoint (défaut: 0)')
    p.add_argument('--steps',       type=int, default=500_000, help='Steps total (phases 1+2)')
    p.add_argument('--no-finetune', action='store_true', help='Sauter la phase 2')
    p.add_argument('--no-waypoint', action='store_true', help='Entraîner sans waypoint (exploration pure)')
    p.add_argument('--chain',       type=int, nargs='+', metavar='WP',
                   help='Enchaîner plusieurs waypoints (ex: --chain 0 1 2)')
    p.add_argument('--n-envs',      type=int, default=12, help='Environnements parallèles')
    p.add_argument('--render',      action='store_true', help='Afficher la fenêtre SDL2')
    p.add_argument('--speed',       type=int, default=0, help='Vitesse émulateur (0=max)')
    return p.parse_args()


def make_env(state: str, waypoint=None, waypoints=None, headless: bool = True,
             speed: int = 0, max_steps: int = 2000, monitor: bool = True, kg=None):
    from src.emulator.pokemon_env import PokemonBlueEnv
    env = PokemonBlueEnv(
        rom_path   = ROM_PATH,
        init_state = state,
        headless   = headless,
        speed      = speed,
        max_steps  = max_steps,
        waypoint   = tuple(waypoint[:3]) if waypoint else None,
        waypoints  = [tuple(w) for w in waypoints] if waypoints else None,
        kg         = kg,   # None → chaque worker charge son propre KG (SubprocVecEnv)
    )
    return Monitor(env) if monitor else env


def run_train(args):
    from src.agent.exploration_agent import ExplorationAgent

    if args.waypoint >= len(WAYPOINTS):
        print(f"[Train] Waypoint {args.waypoint} hors limites (max {len(WAYPOINTS)-1})")
        return

    wp_map, wp_x, wp_y, wp_state, wp_label, wp_max_steps = WAYPOINTS[args.waypoint]
    state    = args.state or wp_state
    save_dir = 'models/rl_checkpoints/'
    os.makedirs(save_dir, exist_ok=True)

    explore_max = wp_max_steps * 5
    steps_p1    = int(args.steps * 0.6)

    print(f"[Train] WP{args.waypoint}: {wp_label}")
    print(f"[Train] Target : map=0x{wp_map:02X}  x={wp_x}  y={wp_y}")
    print(f"[Train] Phase 1: {steps_p1} steps  max/ep={explore_max}  n_envs={args.n_envs}")

    factory_p1 = partial(make_env, state, waypoint=(wp_map, wp_x, wp_y),
                         max_steps=explore_max)
    agent = ExplorationAgent(factory_p1, model_path=args.model, n_envs=args.n_envs)
    p1_path = os.path.join(save_dir, f"wp{args.waypoint}_phase1.zip")
    agent.train(total_timesteps=steps_p1, waypoint_idx=args.waypoint, save_path=p1_path)
    agent.close()
    del agent

    if args.no_finetune:
        return

    steps_p2 = args.steps - steps_p1
    print(f"\n[Train] Phase 2: {steps_p2} steps  max/ep={wp_max_steps}  n_envs={args.n_envs}")

    factory_p2 = partial(make_env, state, waypoint=(wp_map, wp_x, wp_y),
                         max_steps=wp_max_steps)
    agent2 = ExplorationAgent(factory_p2, model_path=p1_path, n_envs=args.n_envs)
    agent2.train(total_timesteps=steps_p2, waypoint_idx=args.waypoint)
    agent2.close()


def run_train_free(args):
    """Entraîne sans waypoint — exploration pure guidée par map discovery + badge reward."""
    from src.agent.exploration_agent import ExplorationAgent

    state    = args.state or INIT_STATE
    save_dir = 'models/rl_checkpoints/'
    os.makedirs(save_dir, exist_ok=True)

    max_ep   = 4000   # budget généreux : l'agent doit traverser plusieurs maps
    steps_p1 = int(args.steps * 0.6)

    print(f"[Free] Exploration pure depuis : {state}")
    print(f"[Free] Reward : map_discovery (+1) + badge (+50) + opp_lvl bonus")
    print(f"[Free] Phase 1 : {steps_p1} steps  max/ep={max_ep}  n_envs={args.n_envs}")

    factory = partial(make_env, state, waypoint=None, waypoints=None, max_steps=max_ep)
    agent   = ExplorationAgent(factory, model_path=args.model, n_envs=args.n_envs)
    p1_path = os.path.join(save_dir, 'free_phase1.zip')
    agent.train(total_timesteps=steps_p1, save_path=p1_path)
    agent.close()
    del agent

    if args.no_finetune:
        return

    steps_p2 = args.steps - steps_p1
    max_ep2  = 2000
    print(f"\n[Free] Phase 2 : {steps_p2} steps  max/ep={max_ep2}")
    factory2 = partial(make_env, state, waypoint=None, waypoints=None, max_steps=max_ep2)
    agent2   = ExplorationAgent(factory2, model_path=p1_path, n_envs=args.n_envs)
    agent2.train(total_timesteps=steps_p2, save_path=os.path.join(save_dir, 'free_final.zip'))
    agent2.close()


def run_train_chain(args):
    from src.agent.exploration_agent import ExplorationAgent

    wps      = [WAYPOINTS[i] for i in args.chain]
    state    = args.state or wps[0][3]
    save_dir = 'models/rl_checkpoints/'
    os.makedirs(save_dir, exist_ok=True)

    wp_tuples  = [(w[0], w[1], w[2]) for w in wps]
    max_steps  = sum(w[5] for w in wps)
    label      = ' → '.join(w[4] for w in wps)
    chain_name = '_'.join(str(i) for i in args.chain)

    print(f"[Chain] {label}")
    print(f"[Chain] max_steps/ep={max_steps}  n_envs={args.n_envs}")

    steps_p1 = int(args.steps * 0.6)
    print(f"\n[Chain] Phase 1 : {steps_p1} steps  max/ep={max_steps * 5}")

    factory_p1 = partial(make_env, state, max_steps=max_steps * 5, waypoints=wp_tuples)
    agent    = ExplorationAgent(factory_p1, model_path=args.model, n_envs=args.n_envs)
    p1_path  = os.path.join(save_dir, f"chain{chain_name}_phase1.zip")
    agent.train(total_timesteps=steps_p1, save_path=p1_path, waypoint_idx=args.chain[0])
    agent.close()
    del agent

    if args.no_finetune:
        return

    steps_p2 = args.steps - steps_p1
    print(f"\n[Chain] Phase 2 : {steps_p2} steps  max/ep={max_steps}")

    factory_p2 = partial(make_env, state, max_steps=max_steps, waypoints=wp_tuples)
    agent2 = ExplorationAgent(factory_p2, model_path=p1_path, n_envs=args.n_envs)
    final_path = os.path.join(save_dir, f"chain{chain_name}_final.zip")
    agent2.train(total_timesteps=steps_p2, save_path=final_path, waypoint_idx=args.chain[0])
    agent2.close()

    print(f"\n[Chain] Modèle final : {final_path}")


def run_inference(args):
    from src.agent.exploration_agent import ExplorationAgent
    from src.agent.battle_agent import BattleAgent
    from src.agent.orchestrator import Orchestrator, GameState

    state    = args.state or INIT_STATE
    headless = not args.render

    wp_start = next((i for i, w in enumerate(WAYPOINTS) if w[3] == state), 0)
    wps      = [(w[0], w[1], w[2]) for w in WAYPOINTS[wp_start:]]

    env      = make_env(state, waypoints=wps, headless=headless, speed=args.speed)
    base_env = env.unwrapped
    obs, _   = env.reset()

    if args.model and os.path.exists(args.model):
        print(f"[Run] Loading model: {args.model}")
        ppo_model = MaskablePPO.load(args.model, env=env)
    else:
        print("[Run] Aucun modèle — MaskablePPO aléatoire")
        ppo_model = MaskablePPO('MlpPolicy', env=env, verbose=0)

    explore_agent = ExplorationAgent.from_model(ppo_model, WAYPOINTS[wp_start:])
    battle_agent  = BattleAgent()
    orch          = Orchestrator(base_env.pyboy, explore_agent, battle_agent)

    print(f"[Run] State : {state}  |  max steps : {MAX_STEPS}")
    print("[Run] Ctrl+C pour arrêter.\n")

    from src.emulator.ram_map import RAM_MAP_ID, RAM_PLAYER_X, RAM_PLAYER_Y, RAM_BADGES

    step = 0
    while step < MAX_STEPS:
        game_state = orch.step(obs)
        obs        = base_env._observe()
        step      += 1

        mid = base_env._r(RAM_MAP_ID)
        x   = base_env._r(RAM_PLAYER_X)
        y   = base_env._r(RAM_PLAYER_Y)

        if game_state == GameState.OVERWORLD:
            if explore_agent.waypoint_reached(mid, x, y):
                explore_agent.advance_waypoint()
                wp = explore_agent.current_waypoint()
                if wp is None:
                    print("[Run] Tous les waypoints atteints !")
                    break
                base_env._waypoints = [tuple(wp[:3])]
                base_env._wp_idx    = 0

        if base_env._r(RAM_BADGES) & 0x01:
            print("[Run] Badge Pierre obtenu ! Mission accomplie.")
            break

        if step % 100 == 0:
            print(f"[Run] step={step:6d}  map=0x{mid:02X}  x={x:3d}  y={y:3d}  {game_state}")

    env.close()
    print(f"[Run] Terminé en {step} steps.")


if __name__ == '__main__':
    if not os.path.exists(ROM_PATH):
        print(f"ROM introuvable : {ROM_PATH}")
        exit(1)

    args = parse_args()
    if args.train and args.chain:
        run_train_chain(args)
    elif args.train and args.no_waypoint:
        run_train_free(args)
    elif args.train:
        run_train(args)
    else:
        run_inference(args)
