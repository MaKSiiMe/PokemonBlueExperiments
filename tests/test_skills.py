"""Skills scriptés (Phase 3) sur des états construits à la main : contrat commun,
lecture des menus, choix d'attaque, politique de dialogue, macros de menu, stratégie."""

import dataclasses

import pytest

from pokeblue.knowledge.gen1_data import CHARMAP, ITEM_IDS, MOVE_IDS, SPECIES, SPECIES_IDS, TYPE_IDS
from pokeblue.knowledge.navigation import Progress, Square
from pokeblue.knowledge.progression import milestone
from pokeblue.orchestrator.core import Orchestrator, goal_key
from pokeblue.orchestrator.strategy import Strategy, hm_to_teach, needs_heal
from pokeblue.skills.base import BaseSkill, Goal, SkillStatus
from pokeblue.skills.battle import BattleSkill, best_move_index, move_score
from pokeblue.skills.dialog import dialog_button
from pokeblue.skills.menu_reader import read_menu
from pokeblue.skills.menus import item_label, weakest_move_slot
from pokeblue.skills.navigation import direction_between
from pokeblue.state import ram_symbols as sym
from pokeblue.state.game_state import Battle, BattleMon, GameState, PartyMon
from pokeblue.state.mode_detector import Mode, detect_mode, steady_bgp
from pokeblue.state.screen import HEIGHT, WIDTH, Screen

# ── Fabriques ─────────────────────────────────────────────────────────────────


def screen(*lines: tuple[int, int, str]) -> bytes:
    """Tilemap vide (décor = tuile 0) avec du texte aux positions (x, y)."""
    tiles = bytearray(WIDTH * HEIGHT)
    for x, y, text in lines:
        for i, ch in enumerate(text):
            tiles[y * WIDTH + x + i] = CHARMAP[ch]
    return bytes(tiles)


def box(x: int, y: int, rows: list[str], width: int) -> list[tuple[int, int, str]]:
    out = [(x, y, "┌" + "─" * width + "┐")]
    for i, row in enumerate(rows):
        out.append((x, y + 1 + i, "│" + row.ljust(width) + "│"))
    out.append((x, y + 1 + len(rows), "└" + "─" * width + "┘"))
    return out


def mon(species="BULBASAUR", level=10, hp=30, max_hp=30, moves=("TACKLE",), pp=None, **kw) -> PartyMon:
    s = SPECIES[SPECIES_IDS[species]]
    ids = tuple(MOVE_IDS[m] for m in moves) + (0,) * (4 - len(moves))
    return PartyMon(species=SPECIES_IDS[species], level=level, hp=hp, max_hp=max_hp, status=0,
                    types=s.types, moves=ids, pp=pp or tuple(20 if m else 0 for m in ids),
                    attack=kw.get("attack", 20), defense=kw.get("defense", 20), speed=20,
                    special=kw.get("special", 20), exp=0)


def battle_mon(species, moves=("TACKLE",), pp=None, hp=30, level=10, **kw) -> BattleMon:
    p = mon(species, level=level, hp=hp, moves=moves, pp=pp, **kw)
    return BattleMon(species=p.species, level=level, hp=hp, max_hp=p.max_hp, status=0, types=p.types,
                     moves=p.moves, pp=p.pp, attack=p.attack, defense=p.defense, speed=20,
                     special=p.special, stat_mods=(7,) * 6)


def state(tilemap: bytes | None = None, party=(), bag=(), battle=None, **kw) -> GameState:
    fields = dict(
        map_id=0, x=5, y=5, facing=sym.SPRITE_FACING_DOWN, party=tuple(party), bag=tuple(bag),
        money=0, badges=0, event_flags=0, pokedex_owned=0, pokedex_seen=0, hidden_objects=0,
        status_flags1=0, player_starter=0, rival_starter=0, walk_bike_surf=0, sprites=(),
        battle=battle, tilemap=tilemap or bytes(WIDTH * HEIGHT), map_pal_offset=0, bgp=steady_bgp(0),
        obp0=0, lcd_on=True, window_y=0, joy_ignore=0,
    )
    fields.update(kw)
    return GameState(**fields)


# ── Contrat des skills ────────────────────────────────────────────────────────


class CountingSkill(BaseSkill):
    name = "counting"
    budget_steps = 3

    def _act(self, s):
        if self.goal.get("fail_at") == self.steps:
            self.fail("échec demandé")
        return "a"


def test_skill_contract_running_timeout_failure_success():
    s = state()
    skill = CountingSkill()
    skill.start(Goal("count"), s)
    assert skill.status(s) is SkillStatus.RUNNING
    for _ in range(3):
        assert skill.act(s) == "a"
    assert skill.status(s) is SkillStatus.TIMEOUT
    skill.start(Goal("count", {"fail_at": 2}), s)          # start() remet tout à zéro
    skill.act(s), skill.act(s)
    assert skill.status(s) is SkillStatus.FAILURE and skill.failure == "échec demandé"
    skill.start(Goal("count"), s)
    skill.done = True
    assert skill.status(s) is SkillStatus.SUCCESS


def test_every_orchestrated_skill_honours_the_contract():
    orch = Orchestrator()
    for skill in [*orch.skills.values(), orch.battle, orch.field_move]:
        assert skill.name and skill.budget_steps > 0
        for method in ("start", "act", "status"):
            assert callable(getattr(skill, method))


def test_goal_key_ignores_milestone_tag():
    a = Goal("navigate", {"map": "ROUTE_1", "milestone": "x"})
    assert goal_key(a) == goal_key(Goal("navigate", {"map": "ROUTE_1"}))


# ── Menus et dialogues ────────────────────────────────────────────────────────


def test_read_menu_and_press_towards():
    tiles = screen(*box(10, 0, ["", "▶POKéDEX", "", " POKéMON", "", " ITEM", "", " EXIT"], 8))
    menu = read_menu(Screen(tiles))
    assert [o.text for o in menu.options] == ["POKéDEX", "POKéMON", "ITEM", "EXIT"]
    assert menu.selected == 0
    assert menu.press_towards(menu.index_of("ITEM")) == "down"
    assert menu.press_towards(0) == "a"


def test_dialog_policy():
    text = box(0, 12, ["", "Do you want to", "", "give a nickname?"], 18)
    yes_no = box(14, 7, ["▶YES", "", " NO"], 4)
    s = state(screen(*text, *yes_no))
    assert detect_mode(s) is Mode.MENU
    assert dialog_button(s) == "down"                   # surnom → NON
    take = box(0, 12, ["", "Take this?"], 18)
    assert dialog_button(state(screen(*take, *yes_no))) == "a"     # sinon OUI
    nurse = box(0, 0, ["▶HEAL", "", " CANCEL"], 7)
    assert dialog_button(state(screen(*nurse))) == "a"
    assert dialog_button(state(screen(*box(0, 12, ["", "Hello!"], 18)))) == "a"   # texte
    assert dialog_button(state()) is None                                       # overworld


def test_item_labels_and_move_to_forget():
    assert item_label(ITEM_IDS["HM_CUT"]) == "HM01"
    assert item_label(ITEM_IDS["HM_SURF"]) == "HM03"
    assert item_label(ITEM_IDS["TM_MEGA_PUNCH"]) == "TM01"
    assert item_label(ITEM_IDS["DOME_FOSSIL"]) == "DOME FOSSIL"
    ivysaur = mon("IVYSAUR", moves=("TACKLE", "GROWL", "LEECH_SEED", "VINE_WHIP"))
    assert weakest_move_slot(ivysaur) in (1, 2)          # une attaque de statut
    assert weakest_move_slot(ivysaur) != 3


# ── Combat ────────────────────────────────────────────────────────────────────


def test_move_choice_uses_type_chart_pp_and_disable():
    user = battle_mon("BULBASAUR", moves=("TACKLE", "VINE_WHIP", "GROWL"))
    geodude = battle_mon("GEODUDE")
    assert move_score(MOVE_IDS["VINE_WHIP"], user, geodude) > move_score(MOVE_IDS["TACKLE"], user, geodude)
    assert best_move_index(user, geodude) == 1
    assert best_move_index(user, geodude, disabled_slot=1) == 0
    no_pp = dataclasses.replace(user, pp=(20, 0, 20, 0))
    assert best_move_index(no_pp, geodude) == 0
    gastly = battle_mon("GASTLY")
    assert move_score(MOVE_IDS["TACKLE"], user, gastly) == 0          # Normal → Spectre
    assert TYPE_IDS["GHOST"] in gastly.types


def test_battle_skill_flees_weak_wild_battles_and_fights_trainers():
    player = battle_mon("BULBASAUR", hp=5, moves=("TACKLE",))
    enemy = battle_mon("PIDGEY")
    main = box(8, 12, ["", "▶FIGHT  PKMN", "", " ITEM   RUN"], 10)
    kw = dict(player=player, enemy=enemy, player_party_index=0, enemy_party_count=1)
    wild = state(screen(*main), party=[mon(hp=5)],
                 battle=Battle(kind=sym.WILD_BATTLE, battle_type=0, **kw))
    skill = BattleSkill()
    skill.start(Goal("battle", {"flee_below": 0.25}), wild)
    assert detect_mode(wild) is Mode.BATTLE_MENU
    assert skill.act(wild) == "down"                    # vers RUN
    trainer = dataclasses.replace(wild, battle=Battle(kind=sym.TRAINER_BATTLE, battle_type=0, **kw))
    assert skill.act(trainer) == "a"                    # FIGHT


# ── Navigation ────────────────────────────────────────────────────────────────


def test_direction_between_handles_ledge_jumps():
    a = Square("ROUTE_1", 5, 5)
    assert direction_between(a, Square("ROUTE_1", 5, 7)) == "down"     # saut de corniche
    assert direction_between(a, Square("ROUTE_1", 4, 5)) == "left"
    assert direction_between(a, Square("ROUTE_1", 6, 6)) is None


# ── Stratégie ─────────────────────────────────────────────────────────────────


def test_needs_heal():
    assert not needs_heal(state(party=[mon(hp=30)]), 0.5)
    assert needs_heal(state(party=[mon(hp=10)]), 0.5)
    assert needs_heal(state(party=[mon(hp=30), mon(hp=0)]), 0.1)
    assert needs_heal(state(party=[mon(moves=("TACKLE",), pp=(0, 0, 0, 0))]), 0.5)   # plus de PP


def test_hm_is_taught_once_badge_obtained():
    bag = [(ITEM_IDS["HM_CUT"], 1)]
    assert hm_to_teach(state(party=[mon()], bag=bag)) is None                 # pas de badge
    badge = 1 << sym.BIT_CASCADEBADGE
    goal = hm_to_teach(state(party=[mon()], bag=bag, badges=badge))
    assert goal == Goal("teach", {"item": "HM_CUT", "move": "CUT", "party_index": 0})
    assert hm_to_teach(state(party=[mon(moves=("CUT",))], bag=bag, badges=badge)) is None


def test_milestone_steps_follow_event_flags():
    from pokeblue.knowledge.gen1_data import EVENT_IDS
    starter = milestone("get_starter")
    assert starter.current_approach(Progress()) == {"map": "ROUTE_1"}
    followed = Progress(events=1 << EVENT_IDS["EVENT_FOLLOWED_OAK_INTO_LAB"])
    assert starter.current_approach(followed)["object"] == "OAKSLAB_BULBASAUR_POKE_BALL"
    surge = milestone("beat_lt_surge")
    assert surge.current_approach(Progress()) == {"skill": "trash_cans"}
    assert surge.current_approach(Progress(events=1 << EVENT_IDS["EVENT_2ND_LOCK_OPENED"])) is None


def test_strategy_starts_with_starter_and_trains_before_brock():
    strategy = Strategy()
    assert strategy.decide(state(map_id=0, x=10, y=16)) == Goal(
        "navigate", {"map": "ROUTE_1", "milestone": "get_starter"})
    m = milestone("beat_brock")
    assert strategy.target_level(state(), m) == 14
    # Jalon courant : le starter (niveau conseillé 5) — on fuit une fois ce niveau atteint.
    assert strategy.battle_goal(state(party=[mon(level=3)]))["flee_wild"] is False
    assert strategy.battle_goal(state(party=[mon(level=5)]))["flee_wild"] is True


@pytest.mark.parametrize("kind", ["navigate", "heal", "train", "teach", "trash_cans", "wander", "dialog"])
def test_every_goal_kind_has_a_skill(kind):
    assert kind in Orchestrator().skills
