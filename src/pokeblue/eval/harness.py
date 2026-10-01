"""Harnais d'évaluation : N runs de l'orchestrateur, rapport JSON + Markdown.

Le rapport donne, pour la baseline (ou toute variante de skills) :
- le jalon atteint par chaque run et la proportion de runs qui atteint chaque jalon ;
- le nombre d'actions par badge (médiane, min, max) ;
- les K.O., blackouts et combats ;
- les échecs par skill (FAILURE / TIMEOUT), avec le savestate de chacun.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml

from pokeblue.knowledge.progression import milestones
from pokeblue.orchestrator.runner import BADGES, RunConfig, run_episode
from pokeblue.orchestrator.strategy import StrategyConfig


@dataclass
class EvalConfig:
    runs: int = 4
    first_seed: int = 0
    workers: int = 4
    run: RunConfig = field(default_factory=RunConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        run = RunConfig(**raw.get("run", {}), strategy=StrategyConfig(**raw.get("strategy", {})))
        return cls(runs=raw.get("runs", 4), first_seed=raw.get("first_seed", 0),
                   workers=raw.get("workers", 4), run=run)


def _run_one(config: RunConfig) -> dict:
    return run_episode(config).to_json()


def run_eval(config: EvalConfig, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    configs = [replace(config.run, seed=seed, log_dir=str(out / f"seed_{seed:03d}"))
               for seed in range(config.first_seed, config.first_seed + config.runs)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, config.workers)) as pool:
        results = list(pool.map(_run_one, configs))
    report = summarize(results)
    report["config"] = asdict(config)
    report["wall_time"] = round(time.perf_counter() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "report.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def _stats(values: list[int]) -> dict | None:
    if not values:
        return None
    return {"median": int(statistics.median(values)), "min": min(values), "max": max(values)}


def summarize(results: list[dict]) -> dict:
    n = len(results)
    order = [m.id for m in milestones()]
    reached = {mid: [r["milestones"][mid] for r in results if mid in r["milestones"]] for mid in order}
    failures = Counter(f"{f['skill']} {f['status']}" for r in results for f in r["failures"])
    examples: dict[str, list[dict]] = {}
    for r in results:
        for f in r["failures"]:
            key = f"{f['skill']} {f['status']}"
            if len(examples.setdefault(key, [])) < 5:
                examples[key].append({k: f.get(k) for k in ("reason", "map", "x", "y", "milestone",
                                                            "savestate")} | {"seed": r["seed"]})
    return {
        "runs": n,
        "results": results,
        "milestones": {mid: {"rate": len(steps) / n, "steps": _stats(steps)}
                       for mid, steps in reached.items() if steps},
        "badges": {b: {"rate": sum(b in r["badges"] for r in results) / n,
                       "steps": _stats([r["badges"][b] for r in results if b in r["badges"]])}
                   for b in BADGES if any(b in r["badges"] for r in results)},
        "badge_count": _stats([len(r["badges"]) for r in results]),
        "kos": _stats([r["kos"] for r in results]),
        "blackouts": _stats([r["blackouts"] for r in results]),
        "failures": dict(failures.most_common()),
        "failure_examples": examples,
        "stop_reasons": dict(Counter(r["stop_reason"] for r in results)),
    }


def _fmt(stats: dict | None) -> str:
    return "—" if stats is None else f"{stats['median']} ({stats['min']}–{stats['max']})"


def to_markdown(report: dict) -> str:
    lines = [f"# Évaluation — {report['runs']} runs", ""]
    lines += ["## Runs", "", "| graine | arrêt | actions | dernier jalon | badges | niveaux | K.O. | blackouts | échecs |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in report["results"]:
        last = max(r["milestones"], key=r["milestones"].get) if r["milestones"] else "—"
        lines.append(f"| {r['seed']} | {r['stop_reason']} | {r['steps']} | {last} | {len(r['badges'])} "
                     f"| {r['final_levels']} | {r['kos']} | {r['blackouts']} | {len(r['failures'])} |")
    lines += ["", "## Jalons", "", "| jalon | runs | actions : médiane (min–max) |", "|---|---|---|"]
    for mid, info in report["milestones"].items():
        lines.append(f"| {mid} | {info['rate']:.0%} | {_fmt(info['steps'])} |")
    lines += ["", "## Badges", "", "| badge | runs | actions : médiane (min–max) |", "|---|---|---|"]
    for badge, info in report["badges"].items():
        lines.append(f"| {badge} | {info['rate']:.0%} | {_fmt(info['steps'])} |")
    lines += ["", f"Badges par run : {_fmt(report['badge_count'])} ; K.O. : {_fmt(report['kos'])} ; "
              f"blackouts : {_fmt(report['blackouts'])} ; arrêts : {report['stop_reasons']}", ""]
    lines += ["## Échecs par skill", "", "| skill | statut | nombre |", "|---|---|---|"]
    for key, count in report["failures"].items():
        skill, status = key.split()
        lines.append(f"| {skill} | {status} | {count} |")
    lines += ["", "Exemples (savestate rejouable avec `pokeblue run --start-state …`) :", ""]
    for key, items in report["failure_examples"].items():
        for f in items[:3]:
            lines.append(f"- {key} — seed {f['seed']}, {f['map']} ({f['x']},{f['y']}), "
                         f"jalon {f['milestone']} : {f['reason']} — `{f['savestate']}`")
    return "\n".join(lines) + "\n"
