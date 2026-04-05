"""
graph.py — Interface de requête sur le graphe de connaissances Gen 1.

Charge le graphe depuis le disque (construit par builder.py) et expose
des méthodes de haut niveau utilisées par BattleAgent et ExplorationAgent.

Usage rapide :
    from src.knowledge.graph import PokemonKnowledgeGraph
    kg = PokemonKnowledgeGraph()                   # charge le graphe

    # Combats
    mult = kg.type_multiplier("water", ["fire", "ground"])  # → 4.0
    idx  = kg.best_move_index([0x37, 0x21, 0x00, 0x00], ["fire", "rock"])  # → 0

    # Exploration
    pokes = kg.encounters_in_zone(0x33)    # Forêt de Jade → ["caterpie", ...]
    evos  = kg.evolutions("caterpie")      # → ["metapod"]
    weak  = kg.weakest_zone_pokemon(0x33, player_types=["water"])  # → "geodude"?
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pyvis.network import Network

import networkx as nx

from src.knowledge.builder import GRAPH_PATH, KnowledgeGraphBuilder
from src.knowledge.gen1_data import (
    RAM_TYPE_BYTE_TO_NAME,
    TYPE_CHART,
    ZONE_MAP_ID,
)

logger = logging.getLogger(__name__)

def visualize_graph(nx_graph, output: str = "docs/pokemon_graph.html") -> None:
    net = Network(
        width="100%",
        height="100vh",
        directed=True,
        bgcolor="#222222",
        font_color="white",
        cdn_resources="remote",
    )
    net.from_nx(nx_graph)
    net.write_html(output, open_browser=False, notebook=False)


class PokemonKnowledgeGraph:
    """Wrappeur de requêtes sur le graphe NetworkX Gen 1.

    Le graphe est chargé une seule fois en mémoire.
    Toutes les méthodes sont sans état (side-effect free).
    """

    def __init__(self, graph_path: Path = GRAPH_PATH, auto_build: bool = True) -> None:
        """
        Args:
            graph_path: Chemin vers le fichier JSON du graphe.
            auto_build: Si True et que le graphe n'existe pas, le construit automatiquement.
        """
        if not graph_path.exists():
            if auto_build:
                logger.info("Graphe absent — lancement du builder (peut prendre quelques minutes)...")
                KnowledgeGraphBuilder(graph_path=graph_path).build()
            else:
                raise FileNotFoundError(
                    f"Graphe introuvable : {graph_path}\n"
                    "Utilisez auto_build=True ou lancez KnowledgeGraphBuilder().build()"
                )
        self._G: nx.DiGraph = KnowledgeGraphBuilder(graph_path=graph_path).load()
        logger.info(
            "Graphe chargé : %d nœuds, %d arêtes",
            self._G.number_of_nodes(), self._G.number_of_edges(),
        )

    # ── Types ─────────────────────────────────────────────────────────────────

    def type_multiplier(self, atk_type: str, def_types: list[str]) -> float:
        """Calcule le multiplicateur de dégâts total pour un type attaquant
        contre un ou plusieurs types défenseurs.

        Gen 1 : les multiplicateurs se combinent par multiplication
        (ex: Water vs Fire/Rock = 2.0 × 2.0 = 4.0).

        Args:
            atk_type:  Nom du type attaquant ("water", "fire", etc.)
            def_types: Liste des types du défenseur (1 ou 2 éléments).

        Returns:
            Multiplicateur final (0.0 / 0.25 / 0.5 / 1.0 / 2.0 / 4.0).
        """
        mult = 1.0
        for def_type in def_types:
            mult *= TYPE_CHART.get(atk_type, {}).get(def_type, 1.0)
        return mult

    def type_multiplier_from_ram(self, atk_type_byte: int, def_type_bytes: list[int]) -> float:
        """Variante qui accepte directement les octets RAM de types.

        Pratique dans BattleAgent qui lit 0xD01F/0xD020 depuis PyBoy.
        """
        atk_name = RAM_TYPE_BYTE_TO_NAME.get(atk_type_byte, "normal")
        def_names = [RAM_TYPE_BYTE_TO_NAME.get(b, "normal") for b in def_type_bytes]
        return self.type_multiplier(atk_name, def_names)

    # ── Moves ─────────────────────────────────────────────────────────────────

    def move_score(
        self,
        move_name: str,
        enemy_types: list[str],
        enemy_hp_pct: float = 1.0,
    ) -> float:
        """Calcule un score heuristique pour un move contre un ennemi donné.

        Critères (par ordre d'importance) :
          1. Multiplicateur de type (priorité maximale)
          2. Base power
          3. Priorité du move (Quick Attack en fin de combat)
          4. Dommage status (malus si ennemi presque KO)

        Args:
            move_name:    Nom PokéAPI du move ("water-gun", "tackle"…).
            enemy_types:  Types du Pokémon ennemi.
            enemy_hp_pct: HP actuels / HP max de l'ennemi (0.0 → 1.0).

        Returns:
            Score flottant — plus c'est haut, meilleur est le move.
        """
        node_id = f"move:{move_name}"
        if not self._G.has_node(node_id):
            return 0.0

        data = self._G.nodes[node_id]
        damage_class: str = data.get("damage_class", "status")

        if damage_class == "status":
            # Les moves de statut sont inutiles si l'ennemi est presque KO
            return -1.0 if enemy_hp_pct < 0.25 else 0.0

        type_name: Optional[str] = data.get("type_name")
        base_power: Optional[int] = data.get("base_power") or 0
        priority: int = data.get("priority", 0)

        # Score de base : multiplicateur de type × puissance normalisée
        mult = self.type_multiplier(type_name, enemy_types) if type_name else 1.0
        score = mult * (base_power / 100.0)

        # Bonus priorité : décisif si ennemi bas PV (Quick Attack finisher)
        if priority > 0 and enemy_hp_pct < 0.3:
            score += 0.5

        return score

    def best_move_index(
        self,
        move_names: list[Optional[str]],
        enemy_type_bytes: list[int],
        enemy_hp_pct: float = 1.0,
    ) -> int:
        """Retourne l'index (0-3) du meilleur move parmi les 4 slots.

        Remplace la logique hardcodée de BattleAgent._best_move_index().
        Utilise les octets RAM de types directement (0xD01F / 0xD020).

        Args:
            move_names:        4 noms de moves (None ou "" = slot vide).
            enemy_type_bytes:  [type1_byte, type2_byte] de l'ennemi.
            enemy_hp_pct:      HP% de l'ennemi (pour bonus Quick Attack).

        Returns:
            Index 0-3 du meilleur move.
        """
        enemy_types = [
            RAM_TYPE_BYTE_TO_NAME.get(b, "normal")
            for b in enemy_type_bytes
        ]

        best_idx, best_score = 0, -999.0
        for i, name in enumerate(move_names):
            if not name:
                continue
            score = self.move_score(name, enemy_types, enemy_hp_pct)
            if score > best_score:
                best_score, best_idx = score, i

        return best_idx

    def moves_for_pokemon(self, pokemon_name: str) -> list[str]:
        """Retourne la liste des moves apprenables par un Pokémon en Red/Blue."""
        node_id = f"pokemon:{pokemon_name}"
        if not self._G.has_node(node_id):
            return []
        return [
            v.split(":", 1)[1]
            for u, v, data in self._G.out_edges(node_id, data=True)
            if data.get("relation") == "LEARNS"
        ]

    # ── Zones / Rencontres ────────────────────────────────────────────────────

    def encounters_in_zone(self, map_id: int) -> list[str]:
        """Retourne les Pokémon sauvages rencontrables dans une zone.

        Args:
            map_id: ID de la map RAM (ex: 0x33 pour Forêt de Jade).

        Returns:
            Liste de noms PokéAPI (ex: ["caterpie", "metapod", "pikachu"]).
        """
        zone_node = f"zone:{map_id:#04x}"
        if not self._G.has_node(zone_node):
            return []
        return [
            u.split(":", 1)[1]
            for u, v, data in self._G.in_edges(zone_node, data=True)
            if data.get("relation") == "FOUND_IN"
        ]

    def next_zones(self, map_id: int) -> list[int]:
        """Retourne les map IDs accessibles depuis la zone courante."""
        zone_node = f"zone:{map_id:#04x}"
        if not self._G.has_node(zone_node):
            return []
        return [
            self._G.nodes[v]["map_id"]
            for _, v, data in self._G.out_edges(zone_node, data=True)
            if data.get("relation") == "LEADS_TO"
        ]

    def zone_type_threat(self, map_id: int, player_types: list[str]) -> float:
        """Évalue la menace d'une zone pour le joueur en fonction de ses types.

        Calcule la somme des multiplicateurs max de chaque ennemi contre
        les types du joueur (plus c'est haut, plus la zone est dangereuse).

        Args:
            map_id:       ID de la map (RAM).
            player_types: Types de l'équipe du joueur.

        Returns:
            Score de menace agrégé (0.0 = aucun danger).
        """
        threat = 0.0
        for pokemon_name in self.encounters_in_zone(map_id):
            pokemon_node = f"pokemon:{pokemon_name}"
            if not self._G.has_node(pokemon_node):
                continue
            enemy_types: list[str] = self._G.nodes[pokemon_node].get("types", [])
            # Multiplicateur max que l'ennemi peut infliger sur un des types joueur
            max_mult = max(
                (self.type_multiplier(et, player_types) for et in enemy_types),
                default=1.0,
            )
            threat += max_mult
        return threat

    # ── Évolutions ────────────────────────────────────────────────────────────

    def evolutions(self, pokemon_name: str) -> list[str]:
        """Retourne les évolutions directes d'un Pokémon."""
        node_id = f"pokemon:{pokemon_name}"
        if not self._G.has_node(node_id):
            return []
        return [
            v.split(":", 1)[1]
            for _, v, data in self._G.out_edges(node_id, data=True)
            if data.get("relation") == "EVOLVES_INTO"
        ]

    def pre_evolutions(self, pokemon_name: str) -> list[str]:
        """Retourne les pré-évolutions d'un Pokémon."""
        node_id = f"pokemon:{pokemon_name}"
        if not self._G.has_node(node_id):
            return []
        return [
            u.split(":", 1)[1]
            for u, _, data in self._G.in_edges(node_id, data=True)
            if data.get("relation") == "EVOLVES_INTO"
        ]

    def evolution_chain(self, pokemon_name: str) -> list[str]:
        """Retourne la chaîne d'évolution complète depuis la forme de base.

        Ex: "charmander" → ["charmander", "charmeleon", "charizard"]
        """
        # Remonter jusqu'à la forme de base
        base = pokemon_name
        visited: set[str] = set()
        while True:
            pre = self.pre_evolutions(base)
            if not pre or pre[0] in visited:
                break
            visited.add(base)
            base = pre[0]

        # Descendre depuis la forme de base
        chain: list[str] = [base]
        current = base
        visited_chain: set[str] = {base}
        while True:
            nexts = self.evolutions(current)
            nexts = [n for n in nexts if n not in visited_chain]
            if not nexts:
                break
            current = nexts[0]
            visited_chain.add(current)
            chain.append(current)

        return chain

    # ── Pokémon ───────────────────────────────────────────────────────────────

    def pokemon_types(self, pokemon_name: str) -> list[str]:
        """Retourne les types d'un Pokémon."""
        node_id = f"pokemon:{pokemon_name}"
        if not self._G.has_node(node_id):
            return []
        return self._G.nodes[node_id].get("types", [])

    def pokemon_stats(self, pokemon_name: str) -> dict[str, int]:
        """Retourne les stats de base d'un Pokémon (hp, attack, defense, speed)."""
        node_id = f"pokemon:{pokemon_name}"
        if not self._G.has_node(node_id):
            return {}
        node = self._G.nodes[node_id]
        return {
            "hp":      node.get("base_hp", 0),
            "attack":  node.get("base_attack", 0),
            "defense": node.get("base_defense", 0),
            "speed":   node.get("base_speed", 0),
        }

    # ── Debug ─────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Retourne un résumé lisible du graphe."""
        kinds: dict[str, int] = {}
        for _, data in self._G.nodes(data=True):
            k = data.get("kind", "unknown")
            kinds[k] = kinds.get(k, 0) + 1

        relations: dict[str, int] = {}
        for _, _, data in self._G.edges(data=True):
            r = data.get("relation", "unknown")
            relations[r] = relations.get(r, 0) + 1

        lines = [
            f"Nœuds  : {self._G.number_of_nodes()} ({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))})",
            f"Arêtes : {self._G.number_of_edges()} ({', '.join(f'{v} {r}' for r, v in sorted(relations.items()))})",
        ]
        return "\n".join(lines)
