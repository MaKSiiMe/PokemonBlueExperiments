"""`pokeblue run` : une partie jouée par l'orchestrateur, avec suivi en direct.

Utile pour rejouer un échec : `pokeblue run --start-state logs/eval/…/failures/…state`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from pokeblue.eval.harness import EvalConfig
from pokeblue.orchestrator.runner import run_episode


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, default=Path("configs/eval/baseline.yaml"))
    p.add_argument("--start-state", type=Path, help="savestate de départ (défaut : configuration)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--no-wait", action="store_true", help="pas d'attente initiale aléatoire")
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--out", type=Path, default=Path("logs/run"))


def run(args: argparse.Namespace) -> int:
    config = EvalConfig.from_yaml(args.config).run
    config = replace(config, seed=args.seed, log_dir=str(args.out / f"seed_{args.seed:03d}"),
                     progress_every=args.progress_every)
    if args.start_state:
        config.start_state = str(args.start_state)
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.no_wait:
        config.max_initial_wait = 0
    result = run_episode(config).to_json()
    failures = result.pop("failures")
    print(json.dumps(result, indent=2))
    print(f"{len(failures)} échec(s) de skill — détails : {args.out}/seed_{args.seed:03d}/result.json")
    return 0
