"""
go_explore.py — Archive sémantique de cellules et wrapper d'environnement.

Implémentation de la stratégie Go-Explore adaptée à Pokémon Bleu :

  Cellule    : triplet (map_id, x, y) — position unique dans le monde du jeu.
  Archive    : dict {cellule → CellEntry} stockant le meilleur savestate
               enregistré pour chaque cellule.
  Exploration: au lieu de toujours repartir de Bourg Palette, l'agent
               sélectionne probabilistiquement une cellule dans l'archive,
               téléporte l'émulateur à cet état, et explore à partir de là.

Stratégie de score pour l'échantillonnage :
  score(c) = 1 / (visits + 1)  ×  recency_weight(c)

  - visits faible → score élevé → cellules rares / frontières préférées.
  - recency_weight → favorise les cellules récemment découvertes
    (proxy des "frontières de l'exploration courante").
  - Le produit équilibre exploration de nouvelles zones et consolidation
    des zones proches de la frontière.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


# ── Entrée d'archive ──────────────────────────────────────────────────────────

@dataclass
class CellEntry:
    """Métadonnées et savestate d'une cellule de l'archive.

    Attributes:
        key          : (map_id, x, y) — identifiant unique de la cellule.
        savestate    : état PyBoy sérialisé en bytes.
        visit_count  : nombre de fois que cette cellule a été visitée.
        discovery_step : numéro de step global auquel la cellule a été vue
                         pour la première fois.
    """
    key:            Tuple[int, int, int]
    savestate:      bytes
    visit_count:    int  = 0
    discovery_step: int  = 0


# ── Archive sémantique ────────────────────────────────────────────────────────

class CellArchive:
    """Base de données de cellules avec stratégie de score pour l'échantillonnage.

    Usage typique :
        archive = CellArchive()
        # Dans la boucle d'entraînement :
        archive.update(map_id, x, y, env.capture_state())
        # Au reset d'un épisode :
        key, state_bytes = archive.sample()
        obs, info = env.reset_from_state(state_bytes)

    Args:
        max_cells      : taille maximale de l'archive (les cellules les plus
                         fréquemment visitées sont évincées si dépassé).
        recency_window : les cellules découvertes dans les N derniers steps
                         globaux reçoivent un bonus de score ×2.
    """

    def __init__(self, max_cells: int = 50_000, recency_window: int = 10_000):
        self._cells: Dict[Tuple[int, int, int], CellEntry] = {}
        self._max_cells     = max_cells
        self._recency_window = recency_window
        self._global_step   = 0

    # ── Mise à jour ───────────────────────────────────────────────────────────

    def update(
        self,
        map_id: int,
        x: int,
        y: int,
        savestate_bytes: bytes,
    ) -> bool:
        """Enregistre ou met à jour une cellule dans l'archive.

        Toujours met à jour le savestate (garde le plus récent),
        incrémente le compteur de visites si la cellule existe déjà.

        Args:
            map_id          : ID de la carte courante.
            x, y            : coordonnées du joueur.
            savestate_bytes : état PyBoy sérialisé.

        Returns:
            True si c'est une nouvelle cellule (découverte), False sinon.
        """
        key = (map_id, x, y)
        self._global_step += 1

        if key not in self._cells:
            # Éviction si l'archive est pleine (supprime la plus visitée)
            if len(self._cells) >= self._max_cells:
                self._evict_one()
            self._cells[key] = CellEntry(
                key=key,
                savestate=savestate_bytes,
                visit_count=0,
                discovery_step=self._global_step,
            )
            return True

        entry = self._cells[key]
        entry.visit_count += 1
        entry.savestate    = savestate_bytes   # savestate le plus récent
        return False

    # ── Échantillonnage ───────────────────────────────────────────────────────

    def sample(self) -> Tuple[Tuple[int, int, int], bytes]:
        """Sélectionne une cellule de l'archive selon le score de probabilité.

        Score = 1/(visits+1) × recency_weight
        Les cellules peu visitées ET récemment découvertes sont préférées.

        Returns:
            (key, savestate_bytes) de la cellule sélectionnée.

        Raises:
            ValueError : si l'archive est vide.
        """
        if not self._cells:
            raise ValueError("CellArchive est vide — aucune cellule à échantillonner.")

        keys    = list(self._cells.keys())
        weights = [self._score(entry) for entry in self._cells.values()]

        selected_key = random.choices(keys, weights=weights, k=1)[0]
        entry = self._cells[selected_key]
        return selected_key, entry.savestate

    def sample_frontier(self, n: int = 10) -> Tuple[Tuple[int, int, int], bytes]:
        """Sélectionne la cellule la moins visitée parmi n cellules récentes.

        Variante plus agressive pour forcer l'exploration des frontières.

        Args:
            n : taille du pool de candidats récents.

        Returns:
            (key, savestate_bytes) de la cellule sélectionnée.
        """
        if not self._cells:
            raise ValueError("CellArchive est vide.")

        # Trie par step de découverte décroissant (les plus récentes en tête)
        recent = sorted(
            self._cells.values(),
            key=lambda e: e.discovery_step,
            reverse=True,
        )[:max(n, 1)]

        # Parmi ces n récentes, prend la moins visitée
        best = min(recent, key=lambda e: e.visit_count)
        return best.key, best.savestate

    # ── Propriétés et utilitaires ─────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Nombre de cellules dans l'archive."""
        return len(self._cells)

    @property
    def unique_maps(self) -> int:
        """Nombre de map_id distincts couverts par l'archive."""
        return len({k[0] for k in self._cells})

    def stats(self) -> dict:
        """Retourne des statistiques de l'archive pour le monitoring."""
        if not self._cells:
            return {'size': 0, 'unique_maps': 0, 'mean_visits': 0.0, 'max_visits': 0}
        visits = [e.visit_count for e in self._cells.values()]
        return {
            'size':        self.size,
            'unique_maps': self.unique_maps,
            'mean_visits': float(np.mean(visits)),
            'max_visits':  int(np.max(visits)),
        }

    # ── Privé ─────────────────────────────────────────────────────────────────

    def _score(self, entry: CellEntry) -> float:
        """Calcule le score de sélection d'une entrée."""
        base   = 1.0 / (entry.visit_count + 1)
        recent = self._global_step - entry.discovery_step < self._recency_window
        return base * (2.0 if recent else 1.0)

    def _evict_one(self):
        """Supprime la cellule avec le plus grand nombre de visites."""
        worst_key = max(self._cells, key=lambda k: self._cells[k].visit_count)
        del self._cells[worst_key]


# ── Wrapper Gymnasium ─────────────────────────────────────────────────────────

class GoExploreWrapper(gym.Wrapper):
    """Wrapper qui remplace les resets standard par des téléportations Go-Explore.

    À chaque reset d'épisode :
      - Avec probabilité (1 - use_archive_prob) : reset normal depuis le
        fichier init_state de l'env (Bourg Palette).
      - Avec probabilité use_archive_prob       : si l'archive contient des
        cellules, on sample une cellule et on charge son savestate via
        env.reset_from_state().

    À chaque step :
      - Capture l'état courant de l'émulateur.
      - Met à jour l'archive avec (map_id, x, y, savestate).

    Compatibilité :
      L'env wrappé doit exposer :
        - capture_state() → bytes
        - reset_from_state(bytes) → (obs, info)
      Ces méthodes sont définies dans PokemonBlueEnv.

    Args:
        env              : instance de PokemonBlueEnv.
        archive          : instance de CellArchive à alimenter.
        use_archive_prob : probabilité de téléportation au reset [0, 1].
                           0.0 = jamais (désactive Go-Explore).
                           0.5 = 50% des épisodes partent d'une cellule archivée.
        capture_every    : capture le savestate tous les N steps seulement
                           (trade-off mémoire / couverture).
    """

    def __init__(
        self,
        env: gym.Env,
        archive: CellArchive,
        use_archive_prob: float = 0.5,
        capture_every: int = 1,
    ):
        super().__init__(env)
        self.archive          = archive
        self.use_archive_prob = use_archive_prob
        self.capture_every    = capture_every
        self._step_counter    = 0
        self._prev_map_id     = -1   # détection des transitions de map

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None) -> Tuple:
        """Reset avec téléportation Go-Explore si l'archive n'est pas vide.

        Stratégie mixte :
          70 % → sample_frontier (cellules de frontière peu visitées)
          30 % → sample pondéré  (exploration équilibrée)
        """
        if self.archive.size > 0 and random.random() < self.use_archive_prob:
            if random.random() < 0.7:
                _, savestate_bytes = self.archive.sample_frontier(n=20)
            else:
                _, savestate_bytes = self.archive.sample()
            obs, info = self.env.reset_from_state(savestate_bytes)
            self._prev_map_id = info.get('map_id', -1)
            return obs, info

        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_map_id = info.get('map_id', -1)
        return obs, info

    def step(self, action) -> Tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)

        map_id     = info['map_id']
        map_change = map_id != self._prev_map_id

        # Capture systématique sur transition de map (ne jamais rater un état frontière)
        # + capture périodique normale
        self._step_counter += 1
        if map_change or self._step_counter % self.capture_every == 0:
            state_bytes = self.env.capture_state()
            self.archive.update(
                map_id=map_id,
                x=info['player_x'],
                y=info['player_y'],
                savestate_bytes=state_bytes,
            )

        self._prev_map_id = map_id
        return obs, reward, terminated, truncated, info
