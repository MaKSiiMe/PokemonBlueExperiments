"""Runner : joue une partie avec l'orchestrateur et mesure ce qui se passe.

Un run part d'un savestate, attend un nombre de frames tiré de la graine (le
générateur aléatoire du jeu avance à chaque frame : rencontres et dégâts changent
d'un run à l'autre), puis enchaîne les actions jusqu'au Champion, au budget
d'actions ou à un blocage. Chaque échec de skill laisse un savestate dans
`<log_dir>/failures/`.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pokeblue.emulator.core import TICKS_PER_ACTION, Emulator
from pokeblue.knowledge.gen1_data import MAPS
from pokeblue.knowledge.navigation import Progress
from pokeblue.knowledge.progression import completed, next_milestone
from pokeblue.orchestrator.core import Orchestrator, SkillEvent
from pokeblue.orchestrator.strategy import Strategy, StrategyConfig
from pokeblue.state import ram_symbols as sym
from pokeblue.state.game_state import GameState

BADGES = ("BOULDERBADGE", "CASCADEBADGE", "THUNDERBADGE", "RAINBOWBADGE",
          "SOULBADGE", "MARSHBADGE", "VOLCANOBADGE", "EARTHBADGE")
MILESTONE_CHECK_EVERY = 10
MAX_FAILURE_STATES = 40


@dataclass
class RunConfig:
    rom: str = "ROMs/PokemonBlue.gb"
    start_state: str = "states/PokemonBlue.gb.state"
    max_steps: int = 50_000
    seed: int = 0
    max_initial_wait: int = 600          # frames
    log_dir: str | None = None
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    progress_every: int = 0              # affiche l'avancement toutes les N actions (0 : jamais)
    stall_steps: int = 20_000            # arrêt si aucun jalon nouveau pendant N actions


@dataclass
class RunResult:
    seed: int
    initial_wait: int
    steps: int = 0
    stop_reason: str = "budget"
    wall_time: float = 0.0
    milestones: dict[str, int] = field(default_factory=dict)   # jalon → action où il est atteint
    badges: dict[str, int] = field(default_factory=dict)
    next_milestone: str | None = None
    final_map: str = ""
    final_levels: list[int] = field(default_factory=list)
    kos: int = 0                    # Pokémon de l'équipe tombés K.O.
    blackouts: int = 0
    battles: dict[str, int] = field(default_factory=lambda: {"wild": 0, "trainer": 0})
    failures: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


class MetricsTracker:
    def __init__(self, result: RunResult) -> None:
        self.result = result
        self.prev: GameState | None = None

    def update(self, state: GameState, step: int) -> None:
        prev, r = self.prev, self.result
        if prev is not None:
            if len(prev.party) == len(state.party):
                r.kos += sum(a.hp > 0 and b.hp == 0 and a.species == b.species
                             for a, b in zip(prev.party, state.party, strict=True))
            if state.all_fainted and not prev.all_fainted:
                r.blackouts += 1
            if state.battle is not None and prev.battle is None:
                r.battles["wild" if state.battle.is_wild else "trainer"] += 1
        if step % MILESTONE_CHECK_EVERY == 0 or prev is None:
            for m in completed(Progress.from_state(state)):
                r.milestones.setdefault(m.id, step)
            for badge in BADGES:
                if state.has_badge(getattr(sym, f"BIT_{badge}")):
                    r.badges.setdefault(badge, step)
        self.prev = state


def run_episode(config: RunConfig) -> RunResult:
    rng = random.Random(config.seed)
    wait = rng.randrange(config.max_initial_wait + 1) if config.max_initial_wait else 0
    result = RunResult(seed=config.seed, initial_wait=wait)
    log_dir = Path(config.log_dir) if config.log_dir else None
    tracker = MetricsTracker(result)
    t0 = time.perf_counter()

    with Emulator(config.rom) as emu:
        emu.load_state(config.start_state)
        emu.tick(wait)

        def on_failure(event: SkillEvent) -> None:
            info = asdict(event)
            if log_dir and len(result.failures) < MAX_FAILURE_STATES:
                path = log_dir / "failures" / f"{event.step:06d}_{event.skill}.state"
                emu.save_state_to(path)
                info["savestate"] = str(path)
            result.failures.append(info)

        orch = Orchestrator(Strategy(config.strategy), on_failure=on_failure, seed=config.seed)
        state = GameState.from_memory(emu.snapshot())
        step = 0
        for step in range(config.max_steps):
            state = GameState.from_memory(emu.snapshot())
            tracker.update(state, step)
            button = orch.step(state)
            if orch.finished:
                result.stop_reason = "champion"
                break
            if orch.stuck:
                result.stop_reason = "stuck"
                break
            if step - max(result.milestones.values(), default=0) > config.stall_steps:
                result.stop_reason = "stalled"
                break
            if button:
                emu.act(button)
            else:
                emu.tick(TICKS_PER_ACTION)
            if config.progress_every and step % config.progress_every == 0:
                goal = orch.active_goal.kind if orch.active_goal else "-"
                print(f"[seed {config.seed}] {step:6d} {MAPS[state.map_id].name:<22} "
                      f"({state.x:2d},{state.y:2d}) niv {[m.level for m in state.party]} "
                      f"{goal:<9} jalons {len(result.milestones)}", flush=True)

        tracker.update(state, step)
        result.steps = step + 1
        result.wall_time = round(time.perf_counter() - t0, 1)
        m = next_milestone(state)
        result.next_milestone = m.id if m else None
        result.final_map = MAPS[state.map_id].name
        result.final_levels = [mon.level for mon in state.party]
        if log_dir:
            emu.save_state_to(log_dir / "final.state")
            (log_dir / "result.json").write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
    return result
