"""Rapport d'évaluation (agrégats et Markdown) sur des résultats fabriqués."""

from pathlib import Path

from pokeblue.eval.harness import EvalConfig, summarize, to_markdown
from pokeblue.orchestrator.runner import RunResult


def _result(seed, milestones, badges, failures=()):
    r = RunResult(seed=seed, initial_wait=0, steps=1000, milestones=milestones, badges=badges,
                  final_levels=[20], kos=seed, blackouts=0)
    r.failures = list(failures)
    return r.to_json()


def test_summary_aggregates_runs():
    fail = {"skill": "navigation", "status": "TIMEOUT", "reason": "budget", "map": "ROUTE_3",
            "x": 1, "y": 2, "milestone": "beat_brock", "savestate": "f.state"}
    results = [
        _result(0, {"get_starter": 10, "beat_brock": 800}, {"BOULDERBADGE": 800}, [fail]),
        _result(1, {"get_starter": 20}, {}),
    ]
    report = summarize(results)
    assert report["milestones"]["get_starter"] == {"rate": 1.0, "steps": {"median": 15, "min": 10, "max": 20}}
    assert report["milestones"]["beat_brock"]["rate"] == 0.5
    assert report["badges"]["BOULDERBADGE"]["steps"]["median"] == 800
    assert report["failures"] == {"navigation TIMEOUT": 1}
    md = to_markdown(report)
    assert "| beat_brock | 50% | 800 (800–800) |" in md
    assert "f.state" in md


def test_eval_config_from_yaml():
    config = EvalConfig.from_yaml(Path("configs/eval/baseline.yaml"))
    assert config.runs >= 1 and config.run.max_steps > 0
    assert 0 < config.run.strategy.heal_below < 1
