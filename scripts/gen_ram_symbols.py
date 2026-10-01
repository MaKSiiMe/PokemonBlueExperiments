#!/usr/bin/env python3
"""Génère `src/pokeblue/state/ram_symbols.py` depuis la table de symboles officielle.

Sources (épinglées dans `pokeblue.knowledge.pokered.source`) :
  - `pokeblue.sym` de la branche `symbols` de pret/pokered : adresses RAM nommées ;
  - `constants/*.asm` du commit `master` correspondant : tailles et offsets de structures.

Usage :
    python scripts/gen_ram_symbols.py           # télécharge (cache .cache/pokered) puis écrit
    python scripts/gen_ram_symbols.py --check   # échoue si le fichier versionné est périmé
"""

from __future__ import annotations

import sys
from pathlib import Path

from _codegen import ROOT, base_parser, write_or_check

from pokeblue.knowledge.pokered.asm import AsmError, constant_name, parse_sym
from pokeblue.knowledge.pokered.source import (
    POKERED_COMMIT,
    SYM_SHA256,
    SYMBOLS_COMMIT,
    fetch_pokered,
    fetch_sym,
    load_constants,
)

OUT = ROOT / "src" / "pokeblue" / "state" / "ram_symbols.py"

# Régions conservées (banque 0 du DMG). La SRAM (A000–BFFF, banquée) contient la
# sauvegarde, pas l'état du jeu en cours : elle est exclue.
REGIONS = {
    "VRAM": (0x8000, 0x9FFF),
    "WRAM": (0xC000, 0xDFFF),  # allow-ram-literal
    "HRAM": (0xFF80, 0xFFFE),  # allow-ram-literal
}

# Constantes de disposition mémoire nécessaires pour lire la RAM, par fichier source.
LAYOUT_CONSTANTS = {
    "constants/pokemon_data_constants.asm": (
        "MON_SPECIES", "MON_HP", "MON_BOX_LEVEL", "MON_STATUS", "MON_TYPE1", "MON_TYPE2",
        "MON_CATCH_RATE", "MON_MOVES", "MON_OTID", "MON_EXP", "MON_HP_EXP", "MON_ATK_EXP",
        "MON_DEF_EXP", "MON_SPD_EXP", "MON_SPC_EXP", "MON_DVS", "MON_PP", "MON_LEVEL",
        "MON_STATS", "MON_MAXHP", "MON_ATK", "MON_DEF", "MON_SPD", "MON_SPC",
        "BOXMON_STRUCT_LENGTH", "PARTYMON_STRUCT_LENGTH", "PARTY_LENGTH", "MONS_PER_BOX",
        "PP_MASK", "PP_UP_MASK",
    ),
    "constants/battle_constants.asm": (
        "NUM_MOVES", "WILD_BATTLE", "TRAINER_BATTLE", "LOST_BATTLE",
        "BATTLE_TYPE_NORMAL", "BATTLE_TYPE_OLD_MAN", "BATTLE_TYPE_SAFARI",
        "SLP_MASK", "PSN", "BRN", "FRZ", "PAR",
    ),
    "constants/ram_constants.asm": (
        "BIT_BOULDERBADGE", "BIT_CASCADEBADGE", "BIT_THUNDERBADGE", "BIT_RAINBOWBADGE",
        "BIT_SOULBADGE", "BIT_MARSHBADGE", "BIT_VOLCANOBADGE", "BIT_EARTHBADGE", "NUM_BADGES",
    ),
    "constants/menu_constants.asm": ("BAG_ITEM_CAPACITY",),
    "constants/text_constants.asm": ("NAME_LENGTH",),
    "constants/pokedex_constants.asm": ("NUM_POKEMON",),
    "constants/event_constants.asm": ("NUM_EVENTS",),
    "constants/sprite_data_constants.asm": (
        "SPRITE_FACING_DOWN", "SPRITE_FACING_UP", "SPRITE_FACING_LEFT", "SPRITE_FACING_RIGHT",
    ),
}

# Tailles des tableaux de bits : macro `flag_array` (macros/ram.asm) = (n + 7) / 8 octets.
FLAG_ARRAY_SIZES = {
    "EVENT_FLAGS_SIZE": ("NUM_EVENTS", "wEventFlags"),
    "BADGE_FLAGS_SIZE": ("NUM_BADGES", "wObtainedBadges"),
    "POKEDEX_FLAGS_SIZE": ("NUM_POKEMON", "wPokedexOwned / wPokedexSeen"),
}


def _region(addr: int) -> str | None:
    for name, (lo, hi) in REGIONS.items():
        if lo <= addr <= hi:
            return name
    return None


def _value(name: str, value: int) -> str:
    return f"0b{value:08b}" if name.endswith("_MASK") else str(value)


def render(sym_path: Path, pokered: Path) -> str:
    symbols = sorted(
        (addr, name)
        for bank, addr, name in parse_sym(sym_path)
        if bank == 0 and _region(addr) and "." not in name
    )
    consts, _ = load_constants(pokered)

    taken: dict[str, str] = {}

    def claim(const: str, origin: str) -> str:
        if const in taken:
            raise AsmError(f"collision de nom {const} : {taken[const]} / {origin}")
        taken[const] = origin
        return const

    lines = [
        '"""Adresses RAM de Pokémon Bleu (US) — FICHIER GÉNÉRÉ, NE PAS MODIFIER.',
        "",
        "Généré par `scripts/gen_ram_symbols.py` depuis pret/pokered :",
        f"  - pokeblue.sym, branche `symbols` @ {SYMBOLS_COMMIT}",
        f"    (sha256 {SYM_SHA256})",
        f"  - constants/*.asm @ {POKERED_COMMIT}",
        "",
        "Chaque adresse porte le nom pokered converti en MAJUSCULES_SOULIGNÉES",
        "(`wEventFlags` → `W_EVENT_FLAGS`). `SYMBOLS` donne le nom d'origine.",
        "C'est la seule source d'adresses RAM autorisée dans le projet.",
        '"""',
        "",
        "# fmt: off",
    ]
    current_region = None
    entries = []
    for addr, name in symbols:
        region = _region(addr)
        if region != current_region:
            lo, hi = REGIONS[region]
            lines += ["", f"# ── {region} ({lo:#06x}–{hi:#06x}) " + "─" * 50]
            current_region = region
        const = claim(constant_name(name), name)
        entries.append((name, const))
        lines.append(f"{const} = 0x{addr:04X}  # {name}")

    lines += ["", "# ── Disposition mémoire (constants/*.asm) " + "─" * 39]
    for rel, names in LAYOUT_CONSTANTS.items():
        lines += ["", f"# {rel}"]
        for name, value in consts.require(*names).items():
            lines.append(f"{claim(name, rel)} = {_value(name, value)}")

    lines += ["", "# Tableaux de bits : macro `flag_array` (macros/ram.asm) = (n + 7) // 8 octets"]
    for const, (count, target) in FLAG_ARRAY_SIZES.items():
        lines.append(f"{claim(const, 'flag_array')} = ({count} + 7) // 8  # {target}")

    lines += ["", "# Nom pokered d'origine → adresse", "SYMBOLS: dict[str, int] = {"]
    lines += [f'    "{name}": {const},' for name, const in entries]
    lines += ["}", "# fmt: on", ""]
    return "\n".join(lines)


def main() -> int:
    args = base_parser(__doc__.splitlines()[1], OUT).parse_args()
    sym = fetch_sym(args.cache_dir)
    pokered = args.pokered or fetch_pokered(args.cache_dir)
    return write_or_check(args.out, render(sym, pokered), args.check)


if __name__ == "__main__":
    sys.exit(main())
