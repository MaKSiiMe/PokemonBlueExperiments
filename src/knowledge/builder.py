"""
builder.py — Construction du graphe de connaissances Pokémon Gen 1.

Fetche la PokéAPI avec un cache disque pour éviter les re-téléchargements.
Construit un nx.DiGraph avec quatre types de nœuds :

  • pokemon   — les 151 Pokémon de Gen 1
  • type      — les 15 types Gen 1
  • move      — tous les moves appris en Red/Blue
  • zone      — les zones du parcours Bourg Palette → Brock

Relations (arêtes orientées) :
  • (pokemon) --HAS_TYPE-->       (type)
  • (pokemon) --LEARNS-->         (move)
  • (pokemon) --FOUND_IN-->       (zone)
  • (pokemon) --EVOLVES_INTO-->   (pokemon)
  • (move)    --IS_TYPE-->        (type)
  • (type)    --SUPER_EFFECTIVE_AGAINST--> (type)   [mult=2.0]
  • (type)    --NOT_EFFECTIVE_AGAINST-->   (type)   [mult=0.5]
  • (type)    --NO_EFFECT_AGAINST-->       (type)   [mult=0.0]
  • (zone)    --LEADS_TO-->       (zone)

Usage :
    from src.knowledge.builder import KnowledgeGraphBuilder
    builder = KnowledgeGraphBuilder()
    G = builder.build()          # fetche + cache + construit
    G = builder.load()           # charge depuis le cache disque
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import networkx as nx
import requests

from src.knowledge.gen1_data import (
    GEN1_TYPES,
    TYPE_CHART,
    ZONE_MAP_ID,
    PRIORITY_MOVES,
)

logger = logging.getLogger(__name__)

# ── Chemins ───────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
CACHE_DIR  = _ROOT / "cache"
GRAPH_PATH = _ROOT / "pokemon_knowledge_graph.json"

# ── PokéAPI ───────────────────────────────────────────────────────────────────
_POKEAPI_BASE = "https://pokeapi.co/api/v2"
_RATE_LIMIT_DELAY = 0.1   # secondes entre requêtes (100 req/s max)
_GEN1_VERSION_GROUP = "red-blue"
_GEN1_POKEMON_COUNT = 151


class PokeAPICache:
    """Cache disque transparent pour la PokéAPI.

    Chaque endpoint est sauvegardé dans un fichier JSON sous CACHE_DIR.
    Un fichier manquant déclenche un fetch HTTP.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "PokemonBlueAI-KnowledgeGraph/1.0"

    def get(self, endpoint: str) -> dict[str, Any]:
        """Retourne les données JSON d'un endpoint PokéAPI (depuis cache ou HTTP)."""
        cache_file = self._dir / (endpoint.replace("/", "_") + ".json")
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        url = f"{_POKEAPI_BASE}/{endpoint}"
        logger.debug("Fetching %s", url)
        response = self._session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(_RATE_LIMIT_DELAY)
        return data

    def get_pokemon(self, dex_number: int) -> dict[str, Any]:
        return self.get(f"pokemon/{dex_number}")

    def get_move(self, move_id: int) -> dict[str, Any]:
        return self.get(f"move/{move_id}")

    def get_location_area(self, slug: str) -> dict[str, Any]:
        return self.get(f"location-area/{slug}")

    def get_evolution_chain(self, chain_id: int) -> dict[str, Any]:
        return self.get(f"evolution-chain/{chain_id}")

    def get_pokemon_species(self, dex_number: int) -> dict[str, Any]:
        return self.get(f"pokemon-species/{dex_number}")


class KnowledgeGraphBuilder:
    """Construit et sauvegarde le graphe de connaissances Gen 1."""

    def __init__(self, cache_dir: Path = CACHE_DIR, graph_path: Path = GRAPH_PATH) -> None:
        self._api = PokeAPICache(cache_dir)
        self._graph_path = graph_path

    # ── API publique ──────────────────────────────────────────────────────────

    def build(self, force_rebuild: bool = False) -> nx.DiGraph:
        """Construit le graphe (ou charge depuis le cache si déjà construit).

        Args:
            force_rebuild: Si True, ignore le cache disque et re-fetche tout.
        """
        if not force_rebuild and self._graph_path.exists():
            logger.info("Graphe trouvé sur disque, chargement depuis %s", self._graph_path)
            return self.load()

        logger.info("Construction du graphe de connaissances Gen 1...")
        G = nx.DiGraph()

        self._add_type_nodes(G)
        self._add_type_edges(G)
        self._add_zone_nodes(G)
        self._add_zone_edges(G)
        self._add_pokemon_and_moves(G)
        self._add_encounter_edges(G)
        self._add_evolution_edges(G)

        self._save(G)
        logger.info(
            "Graphe construit : %d nœuds, %d arêtes → %s",
            G.number_of_nodes(), G.number_of_edges(), self._graph_path,
        )
        return G

    def load(self) -> nx.DiGraph:
        """Charge le graphe depuis le fichier JSON sauvegardé."""
        if not self._graph_path.exists():
            raise FileNotFoundError(
                f"Graphe introuvable : {self._graph_path}\n"
                "Lancez d'abord KnowledgeGraphBuilder().build()"
            )
        data = json.loads(self._graph_path.read_text(encoding="utf-8"))
        return nx.node_link_graph(data, directed=True, multigraph=False)

    # ── Nœuds ─────────────────────────────────────────────────────────────────

    def _add_type_nodes(self, G: nx.DiGraph) -> None:
        for type_name in GEN1_TYPES:
            G.add_node(f"type:{type_name}", kind="type", name=type_name)

    def _add_zone_nodes(self, G: nx.DiGraph) -> None:
        for map_id, info in ZONE_MAP_ID.items():
            node_id = f"zone:{map_id:#04x}"
            G.add_node(
                node_id,
                kind="zone",
                name=info["name"],
                map_id=map_id,
                pokeapi_slug=info["pokeapi_slug"],
            )

    def _add_pokemon_and_moves(self, G: nx.DiGraph) -> None:
        """Fetche les 151 Pokémon depuis PokéAPI et crée les nœuds + arêtes."""
        logger.info("Fetching %d Pokémon depuis PokéAPI...", _GEN1_POKEMON_COUNT)
        move_ids_seen: set[int] = set()

        for dex in range(1, _GEN1_POKEMON_COUNT + 1):
            data = self._api.get_pokemon(dex)
            name = data["name"]
            node_id = f"pokemon:{name}"

            # Filtrer les types Gen 1 (la PokéAPI peut retourner des types modernes
            # pour des formes alternatives — on garde uniquement les types valides)
            types: list[str] = [
                t["type"]["name"]
                for t in data["types"]
                if t["type"]["name"] in GEN1_TYPES
            ]

            G.add_node(
                node_id,
                kind="pokemon",
                name=name,
                dex=dex,
                base_hp=data["stats"][0]["base_stat"],
                base_attack=data["stats"][1]["base_stat"],
                base_defense=data["stats"][2]["base_stat"],
                base_speed=data["stats"][5]["base_stat"],
                types=types,
            )

            # HAS_TYPE
            for type_name in types:
                if type_name in GEN1_TYPES:
                    G.add_edge(node_id, f"type:{type_name}", relation="HAS_TYPE")

            # LEARNS (moves disponibles en Red/Blue uniquement)
            for move_entry in data["moves"]:
                learned_in_rby = any(
                    vgd["version_group"]["name"] == _GEN1_VERSION_GROUP
                    for vgd in move_entry["version_group_details"]
                )
                if not learned_in_rby:
                    continue

                move_url = move_entry["move"]["url"]
                # Extraire l'ID depuis l'URL : "...api/v2/move/33/"
                move_id = int(move_url.rstrip("/").split("/")[-1])
                move_name = move_entry["move"]["name"]
                move_node_id = f"move:{move_name}"

                G.add_edge(node_id, move_node_id, relation="LEARNS")

                if move_id not in move_ids_seen:
                    move_ids_seen.add(move_id)
                    self._add_move_node(G, move_id, move_name)

            if dex % 25 == 0:
                logger.info("  %d/%d Pokémon traités", dex, _GEN1_POKEMON_COUNT)

        logger.info("  %d moves uniques ajoutés au graphe", len(move_ids_seen))

    def _add_move_node(self, G: nx.DiGraph, move_id: int, move_name: str) -> None:
        """Fetche un move et l'ajoute au graphe."""
        try:
            data = self._api.get_move(move_id)
        except requests.HTTPError as exc:
            logger.warning("Move %d (%s) non trouvé : %s", move_id, move_name, exc)
            G.add_node(f"move:{move_name}", kind="move", name=move_name,
                       move_id=move_id, type_name=None, base_power=None,
                       damage_class=None, priority=0)
            return

        type_name = data["type"]["name"]
        # Ignorer les types non-Gen1 (par sécurité)
        if type_name not in GEN1_TYPES:
            type_name = None

        priority = data.get("priority", 0)
        # Certains moves Gen 1 ont une priorité différente dans pokered vs PokéAPI
        # On surcharge avec les données hardcodées si disponibles
        priority = PRIORITY_MOVES.get(move_id, priority)

        node_id = f"move:{move_name}"
        G.add_node(
            node_id,
            kind="move",
            name=move_name,
            move_id=move_id,
            type_name=type_name,
            base_power=data.get("power"),
            damage_class=data["damage_class"]["name"],  # physical/special/status
            priority=priority,
            pp=data.get("pp"),
        )

        if type_name:
            G.add_edge(node_id, f"type:{type_name}", relation="IS_TYPE")

    # ── Arêtes ────────────────────────────────────────────────────────────────

    def _add_type_edges(self, G: nx.DiGraph) -> None:
        """Crée les arêtes d'efficacité entre types depuis TYPE_CHART Gen 1."""
        relation_map = {
            2.0: "SUPER_EFFECTIVE_AGAINST",
            0.5: "NOT_EFFECTIVE_AGAINST",
            0.0: "NO_EFFECT_AGAINST",
        }
        for atk_type, defenses in TYPE_CHART.items():
            for def_type, mult in defenses.items():
                relation = relation_map.get(mult)
                if relation:
                    G.add_edge(
                        f"type:{atk_type}",
                        f"type:{def_type}",
                        relation=relation,
                        multiplier=mult,
                    )

    def _add_zone_edges(self, G: nx.DiGraph) -> None:
        """Crée les arêtes LEADS_TO entre zones dans l'ordre du curriculum."""
        # Ordre topologique Bourg Palette → Brock
        route: list[int] = [0x00, 0x0C, 0x01, 0x0D, 0x0E, 0x33, 0x02, 0x36]
        for i in range(len(route) - 1):
            src = f"zone:{route[i]:#04x}"
            dst = f"zone:{route[i + 1]:#04x}"
            if G.has_node(src) and G.has_node(dst):
                G.add_edge(src, dst, relation="LEADS_TO")

    def _add_encounter_edges(self, G: nx.DiGraph) -> None:
        """Fetche les rencontres sauvages par zone et crée les arêtes FOUND_IN."""
        for map_id, info in ZONE_MAP_ID.items():
            slug = info.get("pokeapi_slug")
            if not slug:
                continue

            zone_node = f"zone:{map_id:#04x}"
            try:
                data = self._api.get_location_area(slug)
            except requests.HTTPError as exc:
                logger.warning("Zone '%s' non trouvée dans PokéAPI : %s", slug, exc)
                continue

            pokemon_seen: set[str] = set()
            for encounter in data.get("pokemon_encounters", []):
                poke_name = encounter["pokemon"]["name"]
                if poke_name in pokemon_seen:
                    continue
                pokemon_seen.add(poke_name)

                # Ne garder que les rencontres Gen 1 (version red ou blue)
                rby_encounter = any(
                    vd["version"]["name"] in ("red", "blue")
                    for vd in encounter.get("version_details", [])
                )
                if not rby_encounter:
                    continue

                pokemon_node = f"pokemon:{poke_name}"
                if G.has_node(pokemon_node):
                    G.add_edge(pokemon_node, zone_node, relation="FOUND_IN")

            logger.debug("Zone %s : %d Pokémon ajoutés", info["name"], len(pokemon_seen))

    def _add_evolution_edges(self, G: nx.DiGraph) -> None:
        """Fetche les chaînes d'évolution et crée les arêtes EVOLVES_INTO."""
        logger.info("Construction des chaînes d'évolution...")
        chain_ids_seen: set[int] = set()

        for dex in range(1, _GEN1_POKEMON_COUNT + 1):
            try:
                species = self._api.get_pokemon_species(dex)
            except requests.HTTPError as exc:
                logger.warning("Espèce %d non trouvée : %s", dex, exc)
                continue

            chain_url = species["evolution_chain"]["url"]
            chain_id = int(chain_url.rstrip("/").split("/")[-1])

            if chain_id in chain_ids_seen:
                continue
            chain_ids_seen.add(chain_id)

            try:
                chain_data = self._api.get_evolution_chain(chain_id)
            except requests.HTTPError as exc:
                logger.warning("Chaîne d'évolution %d non trouvée : %s", chain_id, exc)
                continue

            self._walk_evolution_chain(G, chain_data["chain"])

    def _walk_evolution_chain(self, G: nx.DiGraph, chain_node: dict[str, Any]) -> None:
        """Parcourt récursivement un nœud de chaîne d'évolution."""
        species_name = chain_node["species"]["name"]
        src_node = f"pokemon:{species_name}"

        for evolution in chain_node.get("evolves_to", []):
            evolved_name = evolution["species"]["name"]
            dst_node = f"pokemon:{evolved_name}"
            if G.has_node(src_node) and G.has_node(dst_node):
                G.add_edge(src_node, dst_node, relation="EVOLVES_INTO")
            self._walk_evolution_chain(G, evolution)

    # ── Persistance ───────────────────────────────────────────────────────────

    def _save(self, G: nx.DiGraph) -> None:
        data = nx.node_link_data(G)
        self._graph_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
