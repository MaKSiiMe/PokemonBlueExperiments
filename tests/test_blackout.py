"""test_blackout.py — Le blackout exige que toute l'équipe soit K.O., hors combat."""

from pokeblue.state import ram_symbols as sym


def _set_u16(env, addr, value):
    env.pyboy.memory[addr] = value >> 8
    env.pyboy.memory[addr + 1] = value & 0xFF


def _two_mon_party(env, hp1, hp2):
    env.pyboy.memory[sym.W_IS_IN_BATTLE] = 0
    env.pyboy.memory[sym.W_PARTY_COUNT] = 2
    for hp, mon in ((hp1, 1), (hp2, 2)):
        _set_u16(env, getattr(sym, f"W_PARTY_MON{mon}_MAX_HP"), 20)
        _set_u16(env, getattr(sym, f"W_PARTY_MON{mon}_HP"), hp)


def test_lead_fainted_is_not_a_blackout(env):
    _two_mon_party(env, hp1=0, hp2=12)
    assert not env._blacked_out()


def test_whole_party_fainted_is_a_blackout(env):
    _two_mon_party(env, hp1=0, hp2=0)
    assert env._blacked_out()


def test_no_blackout_during_battle(env):
    _two_mon_party(env, hp1=0, hp2=0)
    env.pyboy.memory[sym.W_IS_IN_BATTLE] = sym.TRAINER_BATTLE
    assert not env._blacked_out()
