"""Baseline scriptée dans l'émulateur (sautés sans ROM ni savestates)."""


from pokeblue.emulator import Emulator
from pokeblue.emulator.core import TICKS_PER_ACTION
from pokeblue.knowledge.gen1_data import MAPS
from pokeblue.orchestrator.runner import RunConfig, run_episode
from pokeblue.skills.base import Goal, SkillStatus
from pokeblue.skills.field import HealSkill
from pokeblue.state.game_state import GameState


def _play(emu: Emulator, skill, max_steps: int) -> GameState:
    s = GameState.from_memory(emu.snapshot())
    for _ in range(max_steps):
        button = skill.act(s)
        if button:
            emu.act(button)
        else:
            emu.tick(TICKS_PER_ACTION)
        s = GameState.from_memory(emu.snapshot())
        if skill.status(s) is not SkillStatus.RUNNING:
            break
    return s


def test_orchestrator_plays_the_opening(rom_path, state_path, tmp_path):
    """Nouvelle partie → starter, combat du rival, colis, Pokédex, sans intervention."""
    config = RunConfig(rom=rom_path, start_state=state_path("PokemonBlue.gb.state"),
                       max_steps=1500, max_initial_wait=0, log_dir=str(tmp_path))
    result = run_episode(config)
    for m in ("get_starter", "rival_oaks_lab", "get_oaks_parcel", "get_pokedex"):
        assert m in result.milestones, (m, result.milestones, result.failures)
    assert result.next_milestone == "beat_brock"
    assert (tmp_path / "result.json").exists() and (tmp_path / "final.state").exists()


def test_heal_skill_reaches_the_nurse(rom_path, state_path):
    with Emulator(rom_path) as emu:
        emu.load_state(state_path("37_pewter_city.state"))
        emu.tick(2)
        skill = HealSkill()
        skill.start(Goal("heal"), GameState.from_memory(emu.snapshot()))
        s = _play(emu, skill, 400)
        assert skill.status(s) is SkillStatus.SUCCESS, skill.failure
        assert MAPS[s.map_id].name == "PEWTER_POKECENTER"
        assert all(mon.hp == mon.max_hp for mon in s.party)
