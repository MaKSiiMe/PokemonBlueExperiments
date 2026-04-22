"""
run_agent.py — Point d'entrée principal.

Objectif : partir du labo du Prof. Chen (Pokédex obtenu) et battre Brock.

Usage :
    python run_agent.py                          # inférence
    python run_agent.py --render                 # avec fenêtre SDL2
    python run_agent.py --train                  # entraîne (2 phases)
    python run_agent.py --train --steps 1000000
    python run_agent.py --train --ram-only       # mode rapide sans CNN
    python run_agent.py --train --curriculum     # curriculum multi-states
    python run_agent.py --model models/rl_checkpoints/final.zip
"""

import argparse
import os
import random
from functools import partial

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

ROM_PATH   = 'ROMs/PokemonBlue.gb'
INIT_STATE = 'states/00_pallet_town.state'
MAX_STEPS  = 100_000

# États du curriculum — du plus facile au plus difficile.
# Chaque fichier doit exister ; les manquants sont ignorés silencieusement.
CURRICULUM_STATES = [
    'states/00_pallet_town.state',   # départ standard
    'states/01_route1.state',        # déjà sur Route 1
    'states/02_viridian.state',      # déjà à Bourg des Eaux
    'states/03_pewter.state',        # devant l'arène de Brock
]

# Poids du tirage par état (plus l'état est avancé, moins il est tiré en début
# d'entraînement ; le poids est uniforme en phase 2).
_CURRICULUM_WEIGHTS_P1 = [0.50, 0.25, 0.15, 0.10]
_CURRICULUM_WEIGHTS_P2 = [0.25, 0.25, 0.25, 0.25]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--state',    default=None,        help='Save state à charger')
    p.add_argument('--model',    default=None,        help='Modèle PPO (.zip)')
    p.add_argument('--train',    action='store_true', help="Entraîner l'agent")
    p.add_argument('--steps',    type=int, default=500_000, help='Steps total (phases 1+2)')
    p.add_argument('--no-finetune', action='store_true', help='Sauter la phase 2')
    p.add_argument('--n-envs',   type=int, default=12,
                   help='Environnements parallèles (défaut: 12 — calibré 12GB WSL2)')
    p.add_argument('--backend',  default='subproc',
                   choices=['dummy', 'subproc'],
                   help='Backend de vectorisation (défaut: subproc)')
    p.add_argument('--compile',  action='store_true',
                   help='Active torch.compile sur la politique (CUDA requis)')
    p.add_argument('--render',   action='store_true', help='Afficher la fenêtre SDL2')
    p.add_argument('--speed',    type=int, default=0, help='Vitesse émulateur (0=max)')
    p.add_argument('--no-go-explore', action='store_true',
                   help='Désactive Go-Explore')
    p.add_argument('--archive-prob', type=float, default=0.6,
                   help='Probabilité de téléportation Go-Explore au reset (défaut: 0.6)')
    p.add_argument('--ram-only', action='store_true',
                   help='Mode RAM-only : observe uniquement le vecteur RAM (16,), '
                        'utilise une MLP au lieu de CNN+GRU — ~10× plus rapide')
    p.add_argument('--curriculum', action='store_true',
                   help='Active le curriculum : tire aléatoirement parmi plusieurs '
                        'savestates à différents stades du jeu (voir CURRICULUM_STATES)')
    return p.parse_args()


def _available_curriculum_states() -> list[str]:
    """Retourne les états du curriculum qui existent sur le disque."""
    return [s for s in CURRICULUM_STATES if os.path.exists(s)]


def make_env(
    state,
    headless:         bool  = True,
    speed:            int   = 0,
    max_steps:        int   = 2000,
    monitor:          bool  = True,
    kg               = None,
    archive          = None,
    use_archive_prob: float = 0.5,
    capture_every:    int   = 10,
    ram_only:         bool  = False,
    curriculum_states: list[str] | None = None,
    curriculum_weights: list[float] | None = None,
):
    """Factory d'environnement.

    Si curriculum_states est fourni, un état est tiré aléatoirement à chaque
    appel (le GoExploreWrapper peut ensuite surcharger le reset).
    """
    from src.emulator.pokemon_env import PokemonBlueEnv

    # Sélection de l'état de départ
    if curriculum_states:
        weights = curriculum_weights or [1.0] * len(curriculum_states)
        chosen_state = random.choices(curriculum_states, weights=weights, k=1)[0]
    else:
        chosen_state = state if isinstance(state, str) else state

    env = PokemonBlueEnv(
        rom_path   = ROM_PATH,
        init_state = chosen_state,
        headless   = headless,
        speed      = speed,
        max_steps  = max_steps,
        kg         = kg,
        ram_only   = ram_only,
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
    from src.agent.go_explore import CellArchive

    state          = args.state or INIT_STATE
    save_dir       = 'models/rl_checkpoints/'
    use_go_explore = not args.no_go_explore
    os.makedirs(save_dir, exist_ok=True)

    # Curriculum : liste des états disponibles
    curriculum_states = None
    if args.curriculum:
        curriculum_states = _available_curriculum_states()
        if len(curriculum_states) <= 1:
            print("[Train] Curriculum activé mais aucun état avancé trouvé dans states/.")
            print("        Crée des états avec : python utils/create_curriculum_states.py")
            print(f"        États attendus : {CURRICULUM_STATES[1:]}")
            curriculum_states = None

    go_kwargs = dict(
        archive          = CellArchive(),
        use_archive_prob = args.archive_prob,
        capture_every    = 10,
    ) if use_go_explore else {}

    # ── Phase 1 — exploration large ───────────────────────────────────────────
    max_ep_p1 = 6000
    steps_p1  = int(args.steps * 0.6)
    print(f"[Train] Objectif    : battre Brock (Badge Pierre)")
    print(f"[Train] Mode        : {'RAM-only MLP' if args.ram_only else 'CNN+GRU'}")
    print(f"[Train] Go-Explore  : {'ON  (prob=' + str(args.archive_prob) + ')' if use_go_explore else 'OFF'}")
    print(f"[Train] Curriculum  : {'ON  (' + str(len(curriculum_states or [])) + ' états)' if curriculum_states else 'OFF'}")
    print(f"[Train] Phase 1     : {steps_p1:,} steps  max/ep={max_ep_p1}  n_envs={args.n_envs}")

    factory_kwargs_p1 = dict(
        max_steps          = max_ep_p1,
        ram_only           = args.ram_only,
        curriculum_states  = curriculum_states,
        curriculum_weights = _CURRICULUM_WEIGHTS_P1 if curriculum_states else None,
        **go_kwargs,
    )
    factory_p1     = partial(make_env, state, **factory_kwargs_p1)
    factory_p1_vid = partial(make_env, state, max_steps=500, monitor=False, ram_only=args.ram_only)
    video_freq_p1  = max(steps_p1 // 3, 10_000)

    agent = ExplorationAgent(
        factory_p1,
        model_path    = args.model,
        n_envs        = args.n_envs,
        backend       = args.backend,
        compile_model = args.compile,
        ram_only      = args.ram_only,
    )
    p1_path = os.path.join(save_dir, 'phase1.zip')
    agent.train(
        total_timesteps   = steps_p1,
        save_path         = p1_path,
        env_factory_video = None if args.ram_only else factory_p1_vid,
        video_freq        = video_freq_p1,
    )
    agent.close()
    del agent

    if args.no_finetune:
        return

    # ── Phase 2 — fine-tune (épisodes plus courts, états plus avancés) ────────
    max_ep_p2 = 3000
    steps_p2  = args.steps - steps_p1
    print(f"\n[Train] Phase 2  : {steps_p2:,} steps  max/ep={max_ep_p2}  n_envs={args.n_envs}")

    factory_kwargs_p2 = dict(
        max_steps          = max_ep_p2,
        ram_only           = args.ram_only,
        curriculum_states  = curriculum_states,
        curriculum_weights = _CURRICULUM_WEIGHTS_P2 if curriculum_states else None,
        **go_kwargs,
    )
    factory_p2 = partial(make_env, state, **factory_kwargs_p2)

    agent2 = ExplorationAgent(
        factory_p2,
        model_path    = p1_path,
        n_envs        = args.n_envs,
        backend       = args.backend,
        compile_model = args.compile,
        ram_only      = args.ram_only,
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

    env      = make_env(state, headless=headless, speed=args.speed,
                        max_steps=MAX_STEPS, ram_only=args.ram_only)
    base_env = env.unwrapped
    obs, _   = env.reset()

    use_random = not (args.model and os.path.exists(args.model))
    if not use_random:
        print(f"[Run] Loading model: {args.model}")
        ppo_model = MaskablePPO.load(args.model, env=env)
        agent = ExplorationAgent.from_model(ppo_model, env=env)
    else:
        print("[Run] Aucun modèle — actions uniformément aléatoires (test de praticabilité)")
        agent = None

    print(f"[Run] State : {state}  |  max steps : {MAX_STEPS}")
    print("[Run] Ctrl+C pour arrêter.\n")

    from src.emulator.ram_map import RAM_MAP_ID, RAM_PLAYER_X, RAM_PLAYER_Y

    step = 0
    try:
        while step < MAX_STEPS:
            masks = base_env.action_masks()
            if use_random:
                valid  = [i for i, m in enumerate(masks) if m]
                action = random.choice(valid)
            else:
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
