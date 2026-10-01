"""Équipes des dresseurs telles que le jeu les construit (data/trainers, engine/battle).

- Les Pokémon d'un dresseur connaissent leurs attaques « par défaut » : attaques de
  niveau 1 puis attaques apprises jusqu'à leur niveau, les plus anciennes chassées
  au-delà de quatre (WriteMonMoves, engine/pokemon/evos_moves.asm).
- Champions d'arène : LONE_MOVES[n° d'arène - 1] place une attaque dans le 3e
  emplacement d'un de leurs Pokémon (engine/battle/read_trainer_party.asm).
- Conseil 4 : TEAM_MOVES place une attaque dans le 3e emplacement du 5e Pokémon.
- Champion (RIVAL3) : Roucarnage reçoit Piqué ; le 6e Pokémon reçoit Méga-Sangsue,
  Déflagration ou Blizzard selon le starter du rival (même fichier, .ChampionRival).
- Rival : chaque rencontre a une équipe de base b ; le script choisit b, b+1 ou b+2
  selon le starter du rival (Carapuce, Bulbizarre, Salamèche), voir `rival_party`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from pokeblue.knowledge.gen1_data import (
    LEARNSETS,
    LONE_MOVES,
    MOVE_IDS,
    SPECIES,
    TEAM_MOVES,
    TRAINER_PARTIES,
)
from pokeblue.knowledge.maps import load_map, map_names

SPECIAL_MOVE_SLOT = 2          # 3e emplacement (wEnemyMon<N>Moves + 2)
TEAM_MOVE_MON = 4              # 5e Pokémon (wEnemyMon5Moves)
# Ordre des variantes d'équipe du rival : STARTER2, STARTER3, STARTER1
# (constants/pokemon_constants.asm : STARTER1 = CHARMANDER, 2 = SQUIRTLE, 3 = BULBASAUR).
RIVAL_STARTER_ORDER = ("SQUIRTLE", "BULBASAUR", "CHARMANDER")
CHAMPION_STARTER_MOVE = {"BULBASAUR": "MEGA_DRAIN", "CHARMANDER": "FIRE_BLAST"}
CHAMPION_DEFAULT_STARTER_MOVE = "BLIZZARD"


@dataclass(frozen=True, slots=True)
class TrainerMon:
    species: int
    level: int
    moves: tuple[int, ...]

    @property
    def name(self) -> str:
        return SPECIES[self.species].name


def default_moves(species: int, level: int) -> tuple[int, ...]:
    """Attaques d'un Pokémon sauvage ou de dresseur au niveau donné (4 emplacements)."""
    moves = list(SPECIES[species].start_moves)
    for learn_level, move in LEARNSETS.get(species, ()):
        if learn_level <= level and move not in moves:
            moves.append(move)
            if len(moves) > 4:
                moves.pop(0)
    return tuple(moves) + (0,) * (4 - len(moves))


@cache
def gym_leader_numbers() -> dict[tuple[str, int], int]:
    """(classe, équipe) du champion de chaque arène → n° d'arène (wGymLeaderNo)."""
    numbers = {}
    for name in map_names():
        data = load_map(name)
        if data.gym_leader_no and data.trainers():
            numbers[data.trainers()[0].trainer] = data.gym_leader_no
    return numbers


def rival_party(base: int, rival_starter: str) -> int:
    """N° d'équipe du rival pour une rencontre de base `base` (voir docstring du module)."""
    return base + RIVAL_STARTER_ORDER.index(rival_starter)


def trainer_team(trainer_class: str, party: int,
                 rival_starter: str | None = None) -> tuple[TrainerMon, ...]:
    """Équipe n° `party` (à partir de 1) d'une classe de dresseur, attaques comprises."""
    team = [
        [species, level, list(default_moves(species, level))]
        for species, level in TRAINER_PARTIES[trainer_class][party - 1]
    ]

    def give(index: int, move: str | int) -> None:
        if 0 <= index < len(team):
            team[index][2][SPECIAL_MOVE_SLOT] = MOVE_IDS[move] if isinstance(move, str) else move

    gym_no = gym_leader_numbers().get((trainer_class, party))
    if gym_no:
        index, move = LONE_MOVES[gym_no - 1]
        give(index, move)
    elif trainer_class in TEAM_MOVES:
        give(TEAM_MOVE_MON, TEAM_MOVES[trainer_class])
    elif trainer_class == "RIVAL3":
        give(0, "SKY_ATTACK")
        starter = rival_starter or RIVAL_STARTER_ORDER[party - 1]
        give(5, CHAMPION_STARTER_MOVE.get(starter, CHAMPION_DEFAULT_STARTER_MOVE))
    return tuple(TrainerMon(s, lv, tuple(m)) for s, lv, m in team)


def max_level(team: tuple[TrainerMon, ...]) -> int:
    return max(mon.level for mon in team)


def starter_name(species_id: int) -> str:
    return SPECIES[species_id].name


__all__ = [
    "TrainerMon",
    "default_moves",
    "gym_leader_numbers",
    "max_level",
    "rival_party",
    "starter_name",
    "trainer_team",
]
