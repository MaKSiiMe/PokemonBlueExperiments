"""
run_agent.py — Point d'entrée principal.

Lance l'agent autonome depuis un save state.
L'orchestrateur lit la RAM à chaque step et délègue au bon sous-agent.

Usage:
    python run_agent.py                                 # inférence depuis Bourg Palette
    python run_agent.py --render                        # avec fenêtre de jeu
    python run_agent.py --state states/07_route1_grass.state
    python run_agent.py --model models/rl_checkpoints/explore_wp0_final.zip

    python run_agent.py --train                         # entraîner waypoint courant (0)
    python run_agent.py --train --waypoint 1            # entraîner waypoint 1 spécifique
    python run_agent.py --train --steps 500000 --waypoint 0
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

ROM_PATH   = 'ROMs/PokemonBlue.gb'
INIT_STATE = 'states/06_pallet_town.state'
MAX_STEPS  = 100_000   # budget total pour l'inférence

# Waypoints curriculum — doit rester en sync avec ExplorationAgent.WAYPOINTS
# Format : (map_id, target_x, target_y, state_file, label, max_steps)
# max_steps : budget d'actions par épisode — doit être court pour les petits espaces
#   (épisodes courts = plus d'épisodes = meilleur signal PPO)
WAYPOINTS = [
    (0x26,  6,  2, 'states/01_chambre.state',               "Chambre → escalier (haut-droite)",  30),
    (0x25,  3,  7, 'states/02_maison_1f.state',             "Maison 1F → porte sud",             50),
    (0x00, 10, 11, 'states/06_pallet_town.state',           "Entrer au Labo Chen",              200),
    (0x52,  7,  6, 'states/04_lab.state',                   "Prendre la Pokéball",              150),
    (0x00, 10,  6, 'states/04_lab.state',                   "Sortir du Labo",                   150),
    (0x12, 10,  5, 'states/06_pallet_town.state',           "Route 1 — vers nord",              400),
    (0x01,  5, 10, 'states/07_route1_grass.state',          "Arriver à Jadielle City",          600),
    (0x01,  7,  3, 'states/11_viridian_center.state',       "Entrer au Poké Mart",              400),
    (0x00, 10, 11, 'states/11_viridian_center.state',       "Retour Labo Chen",                 800),
    (0x13,  5,  5, 'states/22_route2_down.state',           "Route 2",                          800),
    (0x33,  4, 12, 'states/24_viridian_forest_down.state',  "Forêt de Jade",                   1200),
    (0x02, 12,  8, 'states/35_pewter_center_front.state',   "Argenta City",                    1200),
    (0x54,  5,  5, 'states/46_pewter_gym_front.state',      "Arène Argenta → Brock",            800),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--state',       default=None,        help='Save state à charger (override waypoint state)')
    p.add_argument('--model',       default=None,        help='Chemin du modèle PPO (.zip)')
    p.add_argument('--train',       action='store_true', help='Entraîner l\'agent exploration')
    p.add_argument('--waypoint',    type=int, default=0, help='Index du waypoint à entraîner (défaut: 0)')
    p.add_argument('--steps',       type=int, default=500_000, help='Steps d\'entraînement total (phases 1+2)')
    p.add_argument('--no-finetune', action='store_true', help='Ne pas faire la phase 2 (fine-tune)')
    p.add_argument('--chain',       type=int, nargs='+', metavar='WP',
                   help='Enchaîner plusieurs waypoints en un seul épisode (ex: --chain 0 1)')
    p.add_argument('--render',      action='store_true', help='Afficher la fenêtre du jeu')
    p.add_argument('--speed',       type=int, default=0, help='Vitesse émulateur (0=max, 1=normal)')
    return p.parse_args()


def make_env(state: str, waypoint=None, waypoints=None, headless: bool = True,
             speed: int = 0, max_steps: int = 2000, monitor: bool = True):
    from src.emulator.pokemon_env import PokemonBlueEnv
    env = PokemonBlueEnv(
        rom_path   = ROM_PATH,
        init_state = state,
        headless   = headless,
        speed      = speed,
        max_steps  = max_steps,
        waypoint   = tuple(waypoint[:3]) if waypoint else None,
        waypoints  = [tuple(w) for w in waypoints] if waypoints else None,
    )
    if monitor:
        env = Monitor(env)
    return env


def run_train(args):
    from src.agent.exploration_agent import ExplorationAgent

    wp_idx = args.waypoint
    if wp_idx >= len(WAYPOINTS):
        print(f"[Train] Waypoint index {wp_idx} hors limites (max {len(WAYPOINTS)-1})")
        return

    wp_map, wp_x, wp_y, wp_state, wp_label, wp_max_steps = WAYPOINTS[wp_idx]
    state   = args.state or wp_state
    save_dir = "models/rl_checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    # ── Phase 1 : exploration large ───────────────────────────────────────────
    # max_steps élevé pour que la politique random trouve le waypoint par hasard
    explore_max_steps = wp_max_steps * 5
    steps_p1 = int(args.steps * 0.6)

    print(f"[Train] Waypoint {wp_idx}: {wp_label}")
    print(f"[Train] Target:  map=0x{wp_map:02X}  x={wp_x}  y={wp_y}")
    print(f"[Train] Phase 1: {steps_p1} steps  max_steps/ep={explore_max_steps}")

    factory_p1 = lambda: make_env(
        state, waypoint=(wp_map, wp_x, wp_y), max_steps=explore_max_steps
    )
    agent = ExplorationAgent(factory_p1, model_path=args.model)
    phase1_path = os.path.join(save_dir, f"wp{wp_idx}_phase1.zip")
    agent.train(total_timesteps=steps_p1, waypoint_idx=wp_idx,
                save_path=phase1_path)

    if args.no_finetune:
        return

    # ── Phase 2 : fine-tune ───────────────────────────────────────────────────
    # max_steps serré, repart des poids de la phase 1
    steps_p2 = args.steps - steps_p1

    print(f"\n[Train] Phase 2 (fine-tune): {steps_p2} steps  max_steps/ep={wp_max_steps}")

    factory_p2 = lambda: make_env(
        state, waypoint=(wp_map, wp_x, wp_y), max_steps=wp_max_steps
    )
    agent2 = ExplorationAgent(factory_p2, model_path=phase1_path)
    agent2.train(total_timesteps=steps_p2, waypoint_idx=wp_idx)


def run_train_chain(args):
    """Entraîne un modèle unique sur plusieurs waypoints enchaînés en un seul épisode."""
    from src.agent.exploration_agent import ExplorationAgent

    indices = args.chain
    wps     = [WAYPOINTS[i] for i in indices]
    state   = args.state or wps[0][3]   # state de départ = celui du premier waypoint
    save_dir = "models/rl_checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    wp_tuples  = [(w[0], w[1], w[2]) for w in wps]
    max_steps  = sum(w[5] for w in wps)          # budget = somme des budgets individuels
    label      = " → ".join(w[4] for w in wps)
    chain_name = "_".join(str(i) for i in indices)

    print(f"[Chain] Waypoints : {label}")
    print(f"[Chain] State     : {state}")
    print(f"[Chain] max_steps/épisode : {max_steps}")

    # ── Phase 1 : exploration large ──────────────────────────────────────────
    steps_p1 = int(args.steps * 0.6)
    print(f"\n[Chain] Phase 1 : {steps_p1} steps  max_steps/ep={max_steps * 5}")

    factory_p1 = lambda: make_env(
        state, max_steps=max_steps * 5,
        waypoints=wp_tuples
    )
    agent    = ExplorationAgent(factory_p1, model_path=args.model)
    p1_path  = os.path.join(save_dir, f"chain{chain_name}_phase1.zip")
    agent.train(total_timesteps=steps_p1, save_path=p1_path,
                waypoint_idx=indices[0])

    if args.no_finetune:
        return

    # ── Phase 2 : fine-tune ──────────────────────────────────────────────────
    steps_p2 = args.steps - steps_p1
    print(f"\n[Chain] Phase 2 : {steps_p2} steps  max_steps/ep={max_steps}")

    factory_p2 = lambda: make_env(
        state, max_steps=max_steps,
        waypoints=wp_tuples
    )
    agent2 = ExplorationAgent(factory_p2, model_path=p1_path)
    final_path = os.path.join(save_dir, f"chain{chain_name}_final.zip")
    agent2.train(total_timesteps=steps_p2, save_path=final_path,
                 waypoint_idx=indices[0])

    print(f"\n[Chain] Modèle final : {final_path}")


def run_inference(args):
    from src.agent.exploration_agent import ExplorationAgent
    from src.agent.battle_agent import BattleAgent
    from src.agent.orchestrator import Orchestrator, GameState

    state    = args.state or INIT_STATE
    headless = not args.render

    # Détecter le waypoint de départ selon le state fourni
    wp_start = next((i for i, w in enumerate(WAYPOINTS) if w[3] == state), 0)
    wps = [(w[0], w[1], w[2]) for w in WAYPOINTS[wp_start:]]

    env      = make_env(state, waypoints=wps, headless=headless, speed=args.speed)
    base_env = env.unwrapped   # PokemonBlueEnv sous le Monitor
    obs, _   = env.reset()

    # Load PPO model directly into the existing env — no extra PyBoy instance
    if args.model and os.path.exists(args.model):
        print(f"[Run] Loading model: {args.model}")
        ppo_model = PPO.load(args.model, env=env)
    else:
        print("[Run] No model provided — using untrained PPO (random policy)")
        ppo_model = PPO('MlpPolicy', env=env, verbose=0)

    explore_agent = ExplorationAgent.from_model(ppo_model, WAYPOINTS)
    battle_agent  = BattleAgent()
    orch          = Orchestrator(base_env.pyboy, explore_agent, battle_agent)

    print(f"[Run] Starting from {state}  |  max steps: {MAX_STEPS}")

    step = 0
    while step < MAX_STEPS:
        game_state = orch.step(obs)
        obs        = base_env._observe()
        step      += 1

        mid = base_env._r(0xD35E)
        x   = base_env._r(0xD362)
        y   = base_env._r(0xD361)

        if game_state == GameState.OVERWORLD:
            if explore_agent.waypoint_reached(mid, x, y):
                explore_agent.advance_waypoint()
                wp = explore_agent.current_waypoint()
                if wp is None:
                    print("[Run] All waypoints reached!")
                    break
                base_env._waypoints = [tuple(wp[:3])]
                base_env._wp_idx    = 0

        if base_env._r(0xD356) & 0x01:
            print("[Run] Badge Pierre obtained! Mission complete.")
            break

        if step % 1000 == 0:
            print(f"[Run] step={step:6d}  map=0x{mid:02X}  "
                  f"x={x:3d}  y={y:3d}  state={game_state}")

    env.close()
    print(f"[Run] Done in {step} steps.")


if __name__ == '__main__':
    if not os.path.exists(ROM_PATH):
        print(f"ROM introuvable : {ROM_PATH}")
        exit(1)

    args = parse_args()
    if args.train and args.chain:
        run_train_chain(args)
    elif args.train:
        run_train(args)
    else:
        run_inference(args)
