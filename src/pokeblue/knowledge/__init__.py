"""Couche de connaissance hors-ligne, construite depuis pret/pokered.

API de requête :
    next_milestone(state)            jalon suivant de progression.yaml
    required_level(milestone)        niveau conseillé pour un jalon
    path(map_a, map_b, progress)     trajet de cases entre deux cartes
    type_multiplier(type, types)     efficacité Gen 1 d'une attaque
    trainer_team(classe, équipe)     équipe d'un dresseur, attaques comprises

Données : gen1_data (types, attaques, espèces, dresseurs…), maps (222 cartes),
navigation (Progress, Navigator), progression (jalons), trainers.
"""

from pokeblue.knowledge.gen1_data import type_multiplier
from pokeblue.knowledge.navigation import Ability, Navigator, Progress, path
from pokeblue.knowledge.progression import milestones, next_milestone, required_level
from pokeblue.knowledge.trainers import trainer_team

__all__ = [
    "Ability",
    "Navigator",
    "Progress",
    "milestones",
    "next_milestone",
    "path",
    "required_level",
    "trainer_team",
    "type_multiplier",
]
