"""GameState : état du jeu lu en une passe depuis la RAM.

`GameState.from_memory()` décode un `MemorySnapshot` (aucune dépendance à PyBoy) ;
toutes les adresses et tailles viennent de `ram_symbols` (généré depuis pokered).
`GameState.diff(prev)` résume ce qui a changé entre deux états (récompenses, logs).
"""

from __future__ import annotations

from dataclasses import dataclass

from pokeblue.state import ram_symbols as sym
from pokeblue.state.memory import MemorySnapshot

# Modificateurs de stats utilisés (wPlayerMonStatMods) : Attaque … Esquive.
NUM_USED_STAT_MODS = sym.MOD_EVASION + 1


# ── Décodage ──────────────────────────────────────────────────────────────────

def decode_bcd(raw: bytes) -> int:
    """Entier codé en BCD, octet de poids fort en premier (ex. b"\\x01\\x23\\x45" → 12345).

    Raises:
        ValueError: un quartet n'est pas un chiffre décimal.
    """
    value = 0
    for byte in raw:
        hi, lo = byte >> 4, byte & 0x0F
        if hi > 9 or lo > 9:
            raise ValueError(f"octet BCD invalide : {byte:#04x}")
        value = value * 100 + hi * 10 + lo
    return value


def decode_flags(raw: bytes) -> int:
    """Tableau de bits pokered (macro flag_array) → entier : le drapeau i est le bit i.

    Le drapeau i est le bit (i % 8) de l'octet i // 8, d'où l'ordre petit-boutiste.
    """
    return int.from_bytes(raw, "little")


def set_bits(value: int) -> tuple[int, ...]:
    """Indices des bits à 1, par ordre croissant."""
    indices = []
    while value:
        low = value & -value
        indices.append(low.bit_length() - 1)
        value ^= low
    return tuple(indices)


def _u24(raw: bytes) -> int:
    return int.from_bytes(raw, "big")


# ── Structures ────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PartyMon:
    """Un Pokémon de l'équipe (structure party_struct, wPartyMon1..6)."""

    species: int                 # ID interne (gen1_data.SPECIES)
    level: int
    hp: int
    max_hp: int
    status: int                  # bits 0-2 sommeil, PSN/BRN/FRZ/PAR = bits 3/4/5/6
    types: tuple[int, int]
    moves: tuple[int, ...]       # NUM_MOVES octets, 0 = emplacement vide
    pp: tuple[int, ...]          # PP restants (sans les PP Plus)
    attack: int
    defense: int
    speed: int
    special: int
    exp: int

    @property
    def fainted(self) -> bool:
        return self.hp == 0

    @property
    def hp_fraction(self) -> float:
        return self.hp / self.max_hp if self.max_hp else 0.0


@dataclass(frozen=True, slots=True)
class BattleMon:
    """Un Pokémon actif en combat (structure battle_struct : wBattleMon / wEnemyMon)."""

    species: int                 # 0 tant que le Pokémon n'est pas envoyé
    level: int
    hp: int
    max_hp: int
    status: int
    types: tuple[int, int]
    moves: tuple[int, ...]
    pp: tuple[int, ...]
    attack: int
    defense: int
    speed: int
    special: int
    stat_mods: tuple[int, ...]   # Attaque, Défense, Vitesse, Spécial, Précision, Esquive

    @property
    def hp_fraction(self) -> float:
        return self.hp / self.max_hp if self.max_hp else 0.0


@dataclass(frozen=True, slots=True)
class Battle:
    kind: int                    # wIsInBattle : WILD_BATTLE, TRAINER_BATTLE ou LOST_BATTLE
    battle_type: int             # wBattleType : BATTLE_TYPE_NORMAL, _OLD_MAN, _SAFARI
    player: BattleMon
    enemy: BattleMon
    player_party_index: int      # wPlayerMonNumber : Pokémon de l'équipe au combat
    enemy_party_count: int       # wEnemyPartyCount (combats de dresseur)

    @property
    def is_wild(self) -> bool:
        return self.kind == sym.WILD_BATTLE

    @property
    def is_trainer(self) -> bool:
        return self.kind == sym.TRAINER_BATTLE


def _read_party_mon(mem: MemorySnapshot, index: int) -> PartyMon:
    base = sym.W_PARTY_MON1 + index * sym.PARTYMON_STRUCT_LENGTH
    pp_raw = mem.read(base + sym.MON_PP, sym.NUM_MOVES)
    return PartyMon(
        species=mem[base + sym.MON_SPECIES],
        level=mem[base + sym.MON_LEVEL],
        hp=mem.u16(base + sym.MON_HP),
        max_hp=mem.u16(base + sym.MON_MAXHP),
        status=mem[base + sym.MON_STATUS],
        types=(mem[base + sym.MON_TYPE1], mem[base + sym.MON_TYPE2]),
        moves=tuple(mem.read(base + sym.MON_MOVES, sym.NUM_MOVES)),
        pp=tuple(p & sym.PP_MASK for p in pp_raw),
        attack=mem.u16(base + sym.MON_ATK),
        defense=mem.u16(base + sym.MON_DEF),
        speed=mem.u16(base + sym.MON_SPD),
        special=mem.u16(base + sym.MON_SPC),
        exp=_u24(mem.read(base + sym.MON_EXP, 3)),
    )


def _read_battle_mon(mem: MemorySnapshot, prefix: str, mods_addr: int) -> BattleMon:
    def addr(field: str) -> int:
        return getattr(sym, f"{prefix}_{field}")

    return BattleMon(
        species=mem[addr("SPECIES")],
        level=mem[addr("LEVEL")],
        hp=mem.u16(addr("HP")),
        max_hp=mem.u16(addr("MAX_HP")),
        status=mem[addr("STATUS")],
        types=(mem[addr("TYPE1")], mem[addr("TYPE2")]),
        moves=tuple(mem.read(addr("MOVES"), sym.NUM_MOVES)),
        pp=tuple(p & sym.PP_MASK for p in mem.read(addr("PP"), sym.NUM_MOVES)),
        attack=mem.u16(addr("ATTACK")),
        defense=mem.u16(addr("DEFENSE")),
        speed=mem.u16(addr("SPEED")),
        special=mem.u16(addr("SPECIAL")),
        stat_mods=tuple(mem.read(mods_addr, NUM_USED_STAT_MODS)),
    )


# ── État complet ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class GameState:
    # Position
    map_id: int
    x: int
    y: int
    facing: int                          # SPRITE_FACING_DOWN/UP/LEFT/RIGHT
    # Équipe, sac, progression
    party: tuple[PartyMon, ...]
    bag: tuple[tuple[int, int], ...]     # (objet, quantité)
    money: int
    badges: int                          # bit BIT_*BADGE
    event_flags: int                     # bit i = drapeau i (NUM_EVENTS bits)
    pokedex_owned: int                   # bit i = n° de Pokédex i + 1
    pokedex_seen: int
    # Combat (None hors combat)
    battle: Battle | None
    # Affichage : entrées de la détection de mode
    tilemap: bytes                       # wTileMap, SCREEN_WIDTH × SCREEN_HEIGHT tuiles
    map_pal_offset: int                  # wMapPalOffset (cartes sombres)
    bgp: int                             # rBGP
    obp0: int                            # rOBP0
    lcd_on: bool                         # bit B_LCDC_ENABLE de rLCDC
    window_y: int                        # hWY
    joy_ignore: int                      # wJoyIgnore (boutons ignorés pendant les scripts)

    @classmethod
    def from_memory(cls, mem: MemorySnapshot) -> GameState:
        party_count = min(mem[sym.W_PARTY_COUNT], sym.PARTY_LENGTH)
        bag_count = min(mem[sym.W_NUM_BAG_ITEMS], sym.BAG_ITEM_CAPACITY)
        bag_raw = mem.read(sym.W_BAG_ITEMS, 2 * bag_count)
        battle = None
        if mem[sym.W_IS_IN_BATTLE]:
            battle = Battle(
                kind=mem[sym.W_IS_IN_BATTLE],
                battle_type=mem[sym.W_BATTLE_TYPE],
                player=_read_battle_mon(mem, "W_BATTLE_MON", sym.W_PLAYER_MON_STAT_MODS),
                enemy=_read_battle_mon(mem, "W_ENEMY_MON", sym.W_ENEMY_MON_STAT_MODS),
                player_party_index=mem[sym.W_PLAYER_MON_NUMBER],
                enemy_party_count=mem[sym.W_ENEMY_PARTY_COUNT],
            )
        return cls(
            map_id=mem[sym.W_CUR_MAP],
            x=mem[sym.W_X_COORD],
            y=mem[sym.W_Y_COORD],
            facing=mem[sym.W_SPRITE_PLAYER_STATE_DATA1_FACING_DIRECTION],
            party=tuple(_read_party_mon(mem, i) for i in range(party_count)),
            bag=tuple((bag_raw[i], bag_raw[i + 1]) for i in range(0, len(bag_raw), 2)),
            money=decode_bcd(mem.read(sym.W_PLAYER_MONEY, 3)),
            badges=mem[sym.W_OBTAINED_BADGES],
            event_flags=decode_flags(mem.read(sym.W_EVENT_FLAGS, sym.EVENT_FLAGS_SIZE)),
            pokedex_owned=decode_flags(mem.read(sym.W_POKEDEX_OWNED, sym.POKEDEX_FLAGS_SIZE)),
            pokedex_seen=decode_flags(mem.read(sym.W_POKEDEX_SEEN, sym.POKEDEX_FLAGS_SIZE)),
            battle=battle,
            tilemap=mem.read(sym.W_TILE_MAP, sym.SCREEN_AREA),
            map_pal_offset=mem[sym.W_MAP_PAL_OFFSET],
            bgp=mem[sym.R_BGP],
            obp0=mem[sym.R_OBP0],
            lcd_on=bool(mem[sym.R_LCDC] >> sym.B_LCDC_ENABLE & 1),
            window_y=mem[sym.H_WY],
            joy_ignore=mem[sym.W_JOY_IGNORE],
        )

    # ── Requêtes ──────────────────────────────────────────────────────────────

    def flag(self, event: int) -> bool:
        """Drapeau d'événement (indice de gen1_data.EVENTS, ex. EVENT_BEAT_BROCK)."""
        return bool(self.event_flags >> event & 1)

    @property
    def n_flags(self) -> int:
        return self.event_flags.bit_count()

    def has_badge(self, bit: int) -> bool:
        return bool(self.badges >> bit & 1)

    @property
    def n_badges(self) -> int:
        return self.badges.bit_count()

    def owns(self, dex: int) -> bool:
        return bool(self.pokedex_owned >> (dex - 1) & 1)

    def has_seen(self, dex: int) -> bool:
        return bool(self.pokedex_seen >> (dex - 1) & 1)

    def item_count(self, item: int) -> int:
        return sum(qty for item_id, qty in self.bag if item_id == item)

    @property
    def in_battle(self) -> bool:
        return self.battle is not None

    @property
    def party_hp(self) -> int:
        return sum(mon.hp for mon in self.party)

    @property
    def party_max_hp(self) -> int:
        return sum(mon.max_hp for mon in self.party)

    @property
    def all_fainted(self) -> bool:
        return bool(self.party) and self.party_hp == 0

    def diff(self, prev: GameState) -> StateDiff:
        return StateDiff.between(prev, self)


# ── Différences ───────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class StateDiff:
    """Ce qui a changé de `prev` à `cur` (tuples vides / False / 0 si rien)."""

    new_flags: tuple[int, ...]
    cleared_flags: tuple[int, ...]
    map_change: tuple[int, int] | None           # (ancienne carte, nouvelle carte)
    moved: bool
    new_badges: tuple[int, ...]                  # bits BIT_*BADGE
    fainted: tuple[int, ...]                     # indices d'équipe passés à 0 PV
    level_ups: tuple[tuple[int, int, int], ...]  # (indice, ancien niveau, nouveau)
    battle_started: bool
    battle_ended: bool
    player_mon_fainted: bool                     # Pokémon actif K.O. pendant le combat
    enemy_fainted: bool                          # Pokémon ennemi K.O. pendant le combat
    money_delta: int
    item_deltas: tuple[tuple[int, int], ...]     # (objet, variation), variations non nulles
    new_owned: tuple[int, ...]                   # numéros de Pokédex
    new_seen: tuple[int, ...]

    @classmethod
    def between(cls, prev: GameState, cur: GameState) -> StateDiff:
        fainted, level_ups = [], []
        for i, (old, new) in enumerate(zip(prev.party, cur.party, strict=False)):
            if old.species != new.species:
                continue  # équipe réorganisée : pas de comparaison sur cet emplacement
            if old.hp > 0 and new.hp == 0:
                fainted.append(i)
            if new.level > old.level:
                level_ups.append((i, old.level, new.level))

        player_ko = enemy_ko = False
        if prev.battle and cur.battle:
            p_old, p_new = prev.battle.player, cur.battle.player
            e_old, e_new = prev.battle.enemy, cur.battle.enemy
            player_ko = p_old.species == p_new.species and p_old.hp > 0 and p_new.hp == 0
            enemy_ko = (e_old.species, e_old.level) == (e_new.species, e_new.level) \
                and e_old.hp > 0 and e_new.hp == 0

        items: dict[int, int] = {}
        for item, qty in cur.bag:
            items[item] = items.get(item, 0) + qty
        for item, qty in prev.bag:
            items[item] = items.get(item, 0) - qty

        return cls(
            new_flags=set_bits(cur.event_flags & ~prev.event_flags),
            cleared_flags=set_bits(prev.event_flags & ~cur.event_flags),
            map_change=(prev.map_id, cur.map_id) if prev.map_id != cur.map_id else None,
            moved=(prev.map_id, prev.x, prev.y) != (cur.map_id, cur.x, cur.y),
            new_badges=set_bits(cur.badges & ~prev.badges),
            fainted=tuple(fainted),
            level_ups=tuple(level_ups),
            battle_started=prev.battle is None and cur.battle is not None,
            battle_ended=prev.battle is not None and cur.battle is None,
            player_mon_fainted=player_ko,
            enemy_fainted=enemy_ko,
            money_delta=cur.money - prev.money,
            item_deltas=tuple(sorted((i, d) for i, d in items.items() if d)),
            new_owned=tuple(b + 1 for b in set_bits(cur.pokedex_owned & ~prev.pokedex_owned)),
            new_seen=tuple(b + 1 for b in set_bits(cur.pokedex_seen & ~prev.pokedex_seen)),
        )

    def __bool__(self) -> bool:
        return any(getattr(self, field) for field in self.__slots__)

    def describe(self) -> list[str]:
        """Lignes lisibles (noms pokered) pour les logs."""
        from pokeblue.knowledge.gen1_data import DEX_TO_SPECIES, EVENTS, ITEMS, MAPS, SPECIES

        def map_name(map_id: int) -> str:
            return MAPS[map_id].name if map_id in MAPS else f"map {map_id:#04x}"

        def dex_name(dex: int) -> str:
            return SPECIES[DEX_TO_SPECIES[dex]].name if dex in DEX_TO_SPECIES else f"dex {dex}"

        lines = [f"+ {EVENTS.get(f, f'flag {f:#05x}')}" for f in self.new_flags]
        lines += [f"- {EVENTS.get(f, f'flag {f:#05x}')}" for f in self.cleared_flags]
        if self.map_change:
            lines.append(f"carte {map_name(self.map_change[0])} → {map_name(self.map_change[1])}")
        lines += [f"badge {bit}" for bit in self.new_badges]
        lines += [f"K.O. équipe #{i}" for i in self.fainted]
        lines += [f"niveau équipe #{i} : {a} → {b}" for i, a, b in self.level_ups]
        if self.battle_started:
            lines.append("début de combat")
        if self.battle_ended:
            lines.append("fin de combat")
        if self.player_mon_fainted:
            lines.append("Pokémon actif K.O.")
        if self.enemy_fainted:
            lines.append("ennemi K.O.")
        if self.money_delta:
            lines.append(f"argent {self.money_delta:+d}")
        lines += [f"objet {ITEMS.get(i, hex(i))} {d:+d}" for i, d in self.item_deltas]
        lines += [f"capturé {dex_name(d)}" for d in self.new_owned]
        lines += [f"vu {dex_name(d)}" for d in self.new_seen]
        return lines
