"""
src.knowledge — Graphe de connaissances Pokémon Gen 1.

Point d'entrée principal :
    from src.knowledge import PokemonKnowledgeGraph
    kg = PokemonKnowledgeGraph()   # charge ou construit le graphe automatiquement
"""

from src.knowledge.graph import PokemonKnowledgeGraph

__all__ = ["PokemonKnowledgeGraph"]
