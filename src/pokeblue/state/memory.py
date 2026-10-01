"""Instantané de la mémoire lue en une passe (sans dépendance à PyBoy).

Deux plages suffisent à décrire l'état du jeu : la WRAM (variables du jeu, tilemap
de l'écran) et la page haute FF00–FFFF (registres matériels et HRAM).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pokeblue.state import ram_symbols as sym

WRAM_START, WRAM_END = sym.WRAM_START, sym.WRAM_END
HIGH_START, HIGH_END = sym.IO_START, sym.HIGH_END


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    wram: bytes   # WRAM_START .. WRAM_END inclus
    high: bytes   # HIGH_START .. HIGH_END inclus (registres IO + HRAM)

    def __post_init__(self) -> None:
        if len(self.wram) != WRAM_END - WRAM_START + 1:
            raise ValueError(f"WRAM : {len(self.wram)} octets")
        if len(self.high) != HIGH_END - HIGH_START + 1:
            raise ValueError(f"page haute : {len(self.high)} octets")

    @classmethod
    def blank(cls, values: Mapping[int, int] | None = None) -> MemorySnapshot:
        """Mémoire à zéro, avec éventuellement quelques octets fixés (tests)."""
        wram = bytearray(WRAM_END - WRAM_START + 1)
        high = bytearray(HIGH_END - HIGH_START + 1)
        for addr, value in (values or {}).items():
            buf, base = (wram, WRAM_START) if addr <= WRAM_END else (high, HIGH_START)
            buf[addr - base] = value
        return cls(bytes(wram), bytes(high))

    def _locate(self, addr: int) -> tuple[bytes, int]:
        if WRAM_START <= addr <= WRAM_END:
            return self.wram, addr - WRAM_START
        if HIGH_START <= addr <= HIGH_END:
            return self.high, addr - HIGH_START
        raise IndexError(f"adresse hors instantané : {addr:#06x}")

    def __getitem__(self, addr: int) -> int:
        buf, i = self._locate(addr)
        return buf[i]

    def read(self, addr: int, length: int) -> bytes:
        buf, i = self._locate(addr)
        if i + length > len(buf):
            raise IndexError(f"lecture hors instantané : {addr:#06x}+{length}")
        return buf[i:i + length]

    def u16(self, addr: int) -> int:
        """Entier 16 bits big-endian (ordre utilisé par le jeu pour PV, stats…)."""
        hi, lo = self.read(addr, 2)
        return (hi << 8) | lo
