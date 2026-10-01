"""`pokeblue eval` : évalue l'orchestrateur sur N runs depuis le début du jeu.

Écrit `<out>/report.json`, `<out>/report.md` et, par run, `seed_XXX/result.json`,
`final.state` et les savestates des échecs (`failures/`).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pokeblue.eval.harness import EvalConfig, run_eval


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, default=Path("configs/eval/baseline.yaml"))
    p.add_argument("--runs", type=int, help="remplace `runs` de la configuration")
    p.add_argument("--workers", type=int, help="remplace `workers`")
    p.add_argument("--max-steps", type=int, help="remplace `run.max_steps`")
    p.add_argument("--first-seed", type=int, help="remplace `first_seed`")
    p.add_argument("--out", type=Path, help="dossier du rapport (défaut : logs/eval/<date>)")


def run(args: argparse.Namespace) -> int:
    config = EvalConfig.from_yaml(args.config)
    if args.runs is not None:
        config.runs = args.runs
    if args.workers is not None:
        config.workers = args.workers
    if args.first_seed is not None:
        config.first_seed = args.first_seed
    if args.max_steps is not None:
        config.run.max_steps = args.max_steps
    out = args.out or Path("logs/eval") / time.strftime("%Y%m%d-%H%M%S")
    report = run_eval(config, out)
    print((out / "report.md").read_text(encoding="utf-8"))
    print(f"rapport : {out / 'report.md'} ({report['wall_time']} s)")
    return 0
