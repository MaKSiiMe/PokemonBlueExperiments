"""Parsing minimal de l'assembleur RGBDS de pokered.

Couvre le sous-ensemble utilisé par `constants/*.asm` (macros `const_*`, `map_const`,
`add_tm`/`add_hm`, `trainer_const`,
structures `rsreset`/`rb`/`rw`, `DEF ... EQU`) et les tables de données simples
(`db`, macros à arguments). Tout ce qui n'est pas compris est ignoré, sauf les
directives conditionnelles (`IF`, `REPT`…) : elles lèvent `AsmError`, car les ignorer
pourrait produire des valeurs fausses sans le signaler.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable, Iterator
from pathlib import Path


class AsmError(ValueError):
    """Construction RGBDS non supportée ou expression invalide."""


def strip_comment(line: str) -> str:
    """Retire le commentaire `;` (hors chaînes) et les espaces de bord."""
    in_str = False
    for i, ch in enumerate(line):
        if ch == '"':
            in_str = not in_str
        elif ch == ";" and not in_str:
            return line[:i].strip()
    return line.strip()


def logical_lines(path: Path) -> Iterator[str]:
    """Lignes sans commentaire, continuations `\\` jointes, lignes vides omises."""
    pending = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = strip_comment(raw)
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        line = (pending + line).strip()
        pending = ""
        if line:
            yield line


def macro_args(line: str, name: str) -> list[str] | None:
    """Arguments de `line` si c'est un appel à la macro/directive `name`, sinon None."""
    parts = line.split(None, 1)
    if not parts or parts[0] != name:
        return None
    if len(parts) == 1:
        return []
    return [arg.strip() for arg in parts[1].split(",")]


_NAME_PART = re.compile(r"[A-Z]{2,}s(?![a-z])|[A-Z]+[0-9]*(?![a-z])|[A-Z]?[a-z]+[0-9]*|[0-9]+")


def constant_name(symbol: str) -> str:
    """Nom de constante Python d'un symbole pokered : `wPartyMon1HP` → `W_PARTY_MON1_HP`.

    Les sigles au pluriel restent groupés (`wBattleMonDVs` → `W_BATTLE_MON_DVS`).
    """
    parts = [p for chunk in symbol.split("_") for p in _NAME_PART.findall(chunk)]
    if "".join(parts) != symbol.replace("_", ""):
        raise AsmError(f"symbole non convertible : {symbol!r}")
    return "_".join(parts).upper()


def parse_sym(path: Path) -> list[tuple[int, int, str]]:
    """Lit un fichier de symboles rgblink : liste de (banque, adresse, nom).

    Les constantes exportées (`01 NOM`, sans `banque:adresse`) sont ignorées.
    """
    symbols = []
    for line in logical_lines(path):
        location, name = line.split()
        if ":" not in location:
            continue
        bank, addr = location.split(":")
        symbols.append((int(bank, 16), int(addr, 16), name))
    return symbols


# ── Expressions ───────────────────────────────────────────────────────────────
# Priorités de rgbasm(5), de la plus faible à la plus forte. Attention : elles
# diffèrent de Python/C (`& | ^` passent avant `+ -`, et `<<` aussi).


def _div(a: int, b: int) -> int:
    """Division entière tronquée vers zéro, comme rgbasm (et non `//` de Python)."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def _mod(a: int, b: int) -> int:
    return a - b * _div(a, b)


_BINARY_LEVELS: tuple[dict[str, Callable[[int, int], int]], ...] = (
    {"+": operator.add, "-": operator.sub},
    {"&": operator.and_, "|": operator.or_, "^": operator.xor},
    {"<<": operator.lshift, ">>": operator.rshift},
    {"*": operator.mul, "/": _div, "%": _mod},
)
_UNARY = {"-": operator.neg, "+": operator.pos, "~": operator.invert}

_TOKEN = re.compile(
    r"\s*(?:(?P<hex>\$[0-9A-Fa-f_]+)|(?P<bin>%[01_]+)|(?P<dec>[0-9][0-9_]*)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<op><<|>>|[-+*/%&|^~()]))"
)


def _tokenize(expr: str) -> list[tuple[str, str | int]]:
    """Découpe une expression en (genre, valeur) : ("num", int), ("name", str), ("op", str)."""
    tokens: list[tuple[str, str | int]] = []
    pos, expr = 0, expr.strip()
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m or m.end() == pos:
            raise AsmError(f"expression non supportée : {expr!r}")
        pos = m.end()
        kind, text = m.lastgroup, m.group(m.lastgroup)
        text = text.replace("_", "") if kind in ("hex", "bin", "dec") else text
        prev_operand = bool(tokens) and (tokens[-1][0] != "op" or tokens[-1][1] == ")")
        if kind == "bin" and prev_operand:
            # `%` après un opérande est un modulo suivi d'un nombre décimal.
            tokens += [("op", "%"), ("num", int(text[1:], 10))]
        elif kind == "hex":
            tokens.append(("num", int(text[1:], 16)))
        elif kind == "bin":
            tokens.append(("num", int(text[1:], 2)))
        elif kind == "dec":
            tokens.append(("num", int(text, 10)))  # `05` est valide en RGBDS
        else:
            tokens.append((kind, text))
    return tokens


class _ExprParser:
    """Évaluation par montée de priorités ; `lookup` résout les noms."""

    def __init__(self, expr: str, lookup: Callable[[str], int]) -> None:
        self.expr = expr
        self.tokens = _tokenize(expr)
        self.pos = 0
        self.lookup = lookup

    def parse(self) -> int:
        value = self._binary(0)
        if self.pos != len(self.tokens):
            raise AsmError(f"expression non supportée : {self.expr!r}")
        return value

    def _peek_op(self) -> str | None:
        if self.pos < len(self.tokens) and self.tokens[self.pos][0] == "op":
            return str(self.tokens[self.pos][1])
        return None

    def _binary(self, level: int) -> int:
        if level == len(_BINARY_LEVELS):
            return self._unary()
        ops = _BINARY_LEVELS[level]
        value = self._binary(level + 1)
        while (op := self._peek_op()) in ops:
            self.pos += 1
            value = ops[op](value, self._binary(level + 1))
        return value

    def _unary(self) -> int:
        op = self._peek_op()
        if op in _UNARY:
            self.pos += 1
            return _UNARY[op](self._unary())
        if op == "(":
            self.pos += 1
            value = self._binary(0)
            if self._peek_op() != ")":
                raise AsmError(f"parenthèse non fermée : {self.expr!r}")
            self.pos += 1
            return value
        if self.pos >= len(self.tokens):
            raise AsmError(f"expression incomplète : {self.expr!r}")
        kind, value = self.tokens[self.pos]
        self.pos += 1
        if kind == "num":
            return int(value)
        if kind == "name":
            return self.lookup(str(value))
        raise AsmError(f"expression non supportée : {self.expr!r}")


class AsmConstants:
    """Évalue les constantes définies par une suite de fichiers `constants/*.asm`.

    Les fichiers doivent être chargés dans l'ordre de `includes.asm` quand ils
    dépendent les uns des autres (ex. `NUM_STATS` avant les structures).
    """

    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        # Définitions `EQU` non évaluables (dépendance hors des fichiers chargés).
        self.unresolved: dict[str, str] = {}
        self._const_value = 0
        self._const_inc = 1
        self._rs = 0

    def __getitem__(self, name: str) -> int:
        return self.values[name]

    def eval(self, expr: str) -> int:
        """Évalue une expression entière RGBDS dans le contexte courant."""

        def lookup(name: str) -> int:
            if name == "_RS":
                return self._rs
            if name == "const_value":
                return self._const_value
            if name not in self.values:
                raise AsmError(f"constante inconnue {name!r} dans {expr!r}")
            return self.values[name]

        return _ExprParser(expr, lookup).parse()

    def load(self, path: Path) -> list[str]:
        """Charge un fichier ; retourne les noms énumérés (`const`, `map_const`…) dans l'ordre."""
        enumerated: list[str] = []
        in_macro = False
        for line in logical_lines(path):
            word = line.split(None, 1)[0]
            lower = word.lower()
            if in_macro:
                in_macro = lower != "endm"
                continue
            if lower in ("macro", "macro?"):
                in_macro = True
                continue
            if lower in ("if", "elif", "else", "endc", "rept", "for", "endr"):
                raise AsmError(f"{path}: directive conditionnelle non supportée : {line!r}")
            name = self._directive(line, word)
            if name is not None:
                enumerated.append(name)
        return enumerated

    def _set(self, name: str, value: int) -> None:
        self.values[name] = value
        self.unresolved.pop(name, None)

    def require(self, *names: str) -> dict[str, int]:
        """Valeurs des constantes demandées ; `AsmError` si l'une manque."""
        missing = [n for n in names if n not in self.values]
        if missing:
            raise AsmError(f"constantes introuvables : {', '.join(missing)}")
        return {n: self.values[n] for n in names}

    def _directive(self, line: str, word: str) -> str | None:
        args = macro_args(line, word) or []
        if word == "const_def":
            self._const_value = self.eval(args[0]) if args else 0
            self._const_inc = self.eval(args[1]) if len(args) > 1 else 1
        elif word in ("const", "const_export", "shift_const", "map_const"):
            name = args[0]
            value = 1 << self._const_value if word == "shift_const" else self._const_value
            self._set(name, value)
            if word == "map_const":
                self._set(f"{name}_WIDTH", self.eval(args[1]))
                self._set(f"{name}_HEIGHT", self.eval(args[2]))
            self._const_value += self._const_inc
            return name
        elif word == "trainer_const":
            # constants/trainer_constants.asm : const <classe> + OPP_<classe>.
            name = args[0]
            self._set(name, self._const_value)
            self._set(f"OPP_{name}", self.values["OPP_ID_OFFSET"] + self._const_value)
            self._const_value += self._const_inc
            return name
        elif word in ("add_tm", "add_hm"):
            # Macros de constants/item_constants.asm : `const TM_<move>` / `const HM_<move>`.
            name = f"{word[-2:].upper()}_{args[0]}"
            self._set(name, self._const_value)
            self._const_value += self._const_inc
            return name
        elif word == "const_skip":
            self._const_value += self._const_inc * (self.eval(args[0]) if args else 1)
        elif word == "const_next":
            self._const_value = self.eval(args[0])
        elif word == "rsreset":
            self._rs = 0
        elif word == "rsset":
            self._rs = self.eval(args[0])
        elif word == "rb_skip":
            self._rs += self.eval(args[0]) if args else 1
        elif word in ("DEF", "REDEF"):
            self._define(line.split(None, 1)[1])
        return None

    def _define(self, body: str) -> None:
        m = re.match(r"(\w+)\s+(EQUS|EQU|\+=|-=|=|rb|rw|rl)(?:\s+(.*))?$", body, re.IGNORECASE)
        if not m:
            raise AsmError(f"DEF non supporté : {body!r}")
        name, op, expr = m.group(1), m.group(2).lower(), m.group(3) or ""
        if op == "equs":
            return  # chaîne : hors périmètre
        if op in ("equ", "="):
            # Une définition simple n'a pas d'effet de bord : si elle dépend d'un fichier
            # non chargé (ex. hardware.inc), on la laisse absente plutôt que d'échouer.
            try:
                self._set(name, self.eval(expr))
            except AsmError:
                self.values.pop(name, None)
                self.unresolved[name] = expr
        elif op == "+=":
            self._set(name, self.values[name] + self.eval(expr))
        elif op == "-=":
            self._set(name, self.values[name] - self.eval(expr))
        else:
            size = {"rb": 1, "rw": 2, "rl": 4}[op]
            count = self.eval(expr) if expr else 1
            self._set(name, self._rs)
            self._rs += size * count
