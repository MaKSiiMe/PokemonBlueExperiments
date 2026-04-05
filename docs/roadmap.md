# Roadmap — Pokémon Blue AI Agent

> Document de suivi personnel. Mettre à jour le statut de chaque tâche au fur et à mesure.
>
> **Légende :** ✅ Terminé · 🔄 En cours · ⏳ À faire · ❌ Bloqué · 🚫 Abandonné

---

## Vue d'ensemble

| Phase | Description | Statut |
| :--- | :--- | :---: |
| **0** | Infrastructure & environnement Gym | ✅ |
| **1** | Reward shaping + validation premier run | 🔄 |
| **2** | Orchestrateur (State Machine) | ✅ codé / ⏳ validé |
| **3** | Navigation Agent (PPO) — curriculum waypoints | 🔄 |
| **4** | Battle Agent (heuristique) | ✅ codé / ⏳ validé |
| **5** | Run complet end-to-end | ⏳ |
| **6** | Documentation & Portfolio | ⏳ |

> **Note :** « codé » = le code existe. « validé » = tourné et comportement confirmé en jeu.

---

## ✅ Phase 0 — Infrastructure (complète)

> Décision clé : **abandon total de la vision par ordinateur (YOLO/CV)**. Les observations viennent exclusivement de la RAM Game Boy. Plus robuste, plus rapide à entraîner, aucune dépendance fragile.

| Composant | Fichier | Statut |
| :--- | :--- | :---: |
| Wrapper Gymnasium (RAM-only) | `src/emulator/pokemon_env.py` | ✅ |
| Cartographie RAM | `docs/ram_map.md` | ✅ |
| Mappings tileset/sprites | `mapping.json` | ✅ |
| Outil debug visualisation RAM | `src/utils/debug_visualizer.py` | ✅ |
| ~~Pipeline Vision YOLO~~ | *(supprimé)* | 🚫 |

### Observation vector actuel (9 floats, tous en [0, 1])

| Index | Variable | Adresse RAM |
| :---: | :--- | :--- |
| 0 | `player_x` | `0xD362 / 255` |
| 1 | `player_y` | `0xD361 / 255` |
| 2 | `map_id` | `0xD35E / 255` |
| 3 | `direction` | `0xD35D` → {0, 0.33, 0.66, 1} |
| 4 | `hp_pct` | `0xD16C-D / 0xD18C-D` |
| 5 | `battle_status` | `0xD057 / 2` |
| 6 | `waypoint_x` | cible X courante / 255 (0 si map différente) |
| 7 | `waypoint_y` | cible Y courante / 255 (0 si map différente) |
| 8 | `badges_pct` | `popcount(0xD356) / 8` |

---

## ✅ Phase 1 — Reward shaping (implémentée)

> **Fichier :** `src/emulator/pokemon_env.py`

| Composant | Implémentation | Statut |
| :--- | :--- | :---: |
| Distance shaping au waypoint | `(prev_dist - curr_dist) * 0.1` | ✅ |
| Bonus changement de zone (nouvelles maps seulement) | `+1.0` via `_visited_maps` set | ✅ |
| Penalty stuck (tile visit counts, seuil 600) | `-0.05` via `_tile_visits` dict | ✅ |
| Death penalty (HP → 0 en combat) | `-1.0` | ✅ |
| Opponent level reward (fin de combat) | `+opp_lvl * 0.2` | ✅ |
| Black out detection | `terminated = True` | ✅ |
| Waypoints chaînés (épisode multi-objectifs) | `+2.0` bonus intermédiaire | ✅ |
| Two-phase training (exploration → fine-tune) | `run_agent.py::run_train()` | ✅ |

---

## ✅ Phase 2 — Orchestrateur (codé, à valider)

> **Fichier :** `src/agent/orchestrator.py`

| État | Résumé |
| :--- | :--- |
| ✅ Codé | Routing RAM → 4 états : `FADING`, `DIALOG`, `OVERWORLD`, `BATTLE_WILD/TRAINER` |
| ✅ Codé | Auto-skip dialog (press A), wait on fading, délègue battle à `BattleAgent` |
| ⏳ Valider | Tester en jeu : est-ce que les transitions sont bien détectées ? |
| ⏳ Valider | Est-ce que l'auto-skip passe les menus d'intro / de démarrage ? |

```
0xD057 == 0  →  ExplorationAgent (overworld)
0xD057 == 1  →  BattleAgent (sauvage)
0xD057 == 2  →  BattleAgent (dresseur)
0xD13F != 0  →  Attendre (fade)
0xD11C != 0  →  Auto-skip (dialog → press A)
```

---

## 🔄 Phase 3 — Navigation Agent PPO (curriculum waypoints)

> **Fichiers :** `src/agent/exploration_agent.py`, `run_agent.py`
>
> PPO SB3 `MlpPolicy` en place. Curriculum de 13 waypoints défini. Entraînement two-phase automatisé. **À faire : entraîner tous les waypoints avec la nouvelle obs 9-float.**

### Curriculum — 22 waypoints (Bourg Palette → Badge Pierre)

> Positions marquées `~` = estimées, à affiner si l'agent ne converge pas.

| # | Label | Map cible | Target (x, y) | State de départ | Budget/ep |
| :---: | :--- | :---: | :---: | :--- | :---: |
| 0 | Chambre → escalier | `0x26` | (6, 2) | `01_chambre` | 30 |
| 1 | Maison 1F → sortir | `0x25` | (3, 7) | `02_maison_1f` | 50 |
| 2 | Bourg Palette → Route 1 | `0x00` | (11, 1)~ | `06_pallet_town` | 200 |
| 6 | Route 1 → checkpoint PNJ | `0x0C` | (7, 26) | `07_route1_grass` | 150 |
| 7 | Route 1 → zone corniches | `0x0C` | (7, 7) | `08_route1_pnj1` | 200 |
| 8 | Route 1 → Jadielle City | `0x0C` | (7, 1)~ | `09_route1_ledges` | 100 |
| 9 | Jadielle → Route 2 nord | `0x01` | (19, 1)~ | `14_viridian_up` | 200 |
| 8 | Route 2 → Forêt de Jade | `0x0D` | (6, 1)~ | `22_route2_down` | 1500 |
| 9 | Forêt → zone 1 | `0x33` | (26, 41) | `24_viridian_forest_down` | 200 |
| 10 | Forêt → zone 2 | `0x33` | (26, 25) | `26_viridian_forest_battle1` | 200 |
| 11 | Forêt → zone 4 | `0x33` | (26, 9) | `27_viridian_forest_grass1` | 200 |
| 12 | Forêt → zone 5 | `0x33` | (6, 15) | `29_viridian_forest_item2` | 300 |
| 13 | Forêt → zone 6 | `0x33` | (7, 25) | `30_viridian_forest_grass2` | 150 |
| 14 | Forêt → zone 7 | `0x33` | (1, 16) | `31_viridian_forest_item3` | 150 |
| 15 | Forêt → sortie nord | `0x33` | (1, 1)~ | `32_viridian_forest_battle3` | 150 |
| 16 | Argenta → Arène | `0x02` | (16, 16) | `35_pewter_center_front` | 300 |
| 17 | Arène → Pierre (Brock) | `0x36` | (5, 5)~ | `47_pewter_gym` | 200 |

### Plan d'entraînement

| Tâche | Statut |
| :--- | :---: |
| Réentraîner WP0 + WP1 (obs 9-float) | ⏳ |
| Chaîner WP0+1 en un seul épisode (`--chain 0 1`) | ⏳ |
| Entraîner WP2-4 (Labo Chen) | ⏳ |
| Entraîner WP5-8 (Route 1 → Jadielle aller-retour) | ⏳ |
| Entraîner WP9-12 (Route 2 → Forêt → Argenta → Arène) | ⏳ |

### Optimisations envisagées

| Tâche | Statut | Notes |
| :--- | :---: | :--- |
| Passer à `n_envs=4` (SubprocVecEnv) pour paralléliser | ⏳ | Après validation sur 1 env |

---

## ✅ Phase 4 — Battle Agent (codé, à valider)

> **Fichier :** `src/agent/battle_agent.py`
>
> Approche heuristique pure (pas de RL) — lit la RAM pour choisir la meilleure attaque.

| État | Résumé |
| :--- | :--- |
| ✅ Codé | Lecture HP joueur/ennemi, types ennemis, PP des attaques |
| ✅ Codé | Sélection du move avec type advantage (table Gen 1 partielle) |
| ✅ Codé | Utilisation Potion si HP < 30% |
| ✅ Codé | Navigation menus combat Gen 1 (séquence de boutons) |
| ⏳ Valider | Tester en combat réel (sauvage + dresseur) |
| ⏳ Compléter | Ajouter plus de moves dans `MOVE_TYPES` si nécessaire |

> **Pour Brock :** Pierre et Onix sont faibles Eau/Plante/Combat. Bulbizarre et Carapuce ont un avantage naturel. Salamèche devra utiliser des attaques Normales.

---

## 🔄 Phase 4.5 — Knowledge Graph & Intelligence Tactique

> **Fichiers :** `src/knowledge/` — graphe NetworkX construit depuis PokéAPI, mis en cache sur disque.
>
> Base de données relationnelle de 337 nœuds (151 Pokémon, 163 moves, 15 types, 8 zones) et 4 397 arêtes. Sert de couche de connaissance partagée entre tous les agents.

### Réalisé

| Composant | Fichier | Statut |
| :--- | :--- | :---: |
| Type chart Gen 1 hardcodé (quirks : Ghost→Psy=0, Bug→Psy=2x…) | `gen1_data.py` | ✅ |
| Builder PokéAPI avec cache disque | `builder.py` | ✅ |
| Interface de requête (type_multiplier, encounters, evolutions…) | `graph.py` | ✅ |
| Intégration BattleAgent (remplace TYPE_CHART binaire) | `battle_agent.py` | ✅ |
| Visualisation interactive (GitHub Pages) | `docs/pokemon_graph.html` | ✅ |

### Chantiers prioritaires suivants

#### 1. Action Masking — accélérateur d'entraînement RL

> L'IA RL passe 90% de son temps à tester des actions inutiles. Le masque les élimine avant le choix du réseau.

Avant que le modèle PPO choisisse une action, appliquer un masque binaire généré par le graphe :
- Si l'adversaire est un Fantominus → attaques Normales masquées (multiplicateur = 0.0)
- Si un slot n'a plus de PP → désactivé

Résultat attendu : **temps d'entraînement divisé par ~10** sur les combats.

| Tâche | Statut |
| :--- | :---: |
| Générer le masque d'actions depuis `type_multiplier_from_ram()` | ✅ |
| Brancher le masque dans `MaskablePPO` (sb3-contrib) | ✅ |
| Valider que l'entropie de la policy diminue plus vite | ⏳ |

#### 2. Augmentation de l'Observation — vecteur d'état enrichi

Enrichir le vecteur d'observation 9-float avec un `[Vecteur_KG]` :

$$S = [RAM_{brute}] + [Vecteur_{KG}]$$

Le `Vecteur_KG` cible (3 floats supplémentaires) :

| Index | Valeur | Source |
| :---: | :--- | :--- |
| 9 | Avantage de type actuel (-1 / 0 / 1) | `type_multiplier_from_ram()` |
| 10 | L'ennemi peut-il évoluer ? (0 ou 1) | `evolutions()` |
| 11 | Dangerosité de la zone courante | `zone_type_threat()` |

| Tâche | Statut |
| :--- | :---: |
| Ajouter les 3 floats KG dans `pokemon_env.py::_get_obs()` | ✅ |
| Mettre à jour `observation_space` (9 → 12 floats) | ✅ |
| Ré-entraîner et comparer les courbes de reward | ⏳ |

#### 3. Navigation Stratégique — pathfinding via ZoneNodes

Remplacer la "marche aléatoire" par un itinéraire calculé sur le graphe :

- `nx.shortest_path(G, "zone:0x00", "zone:0x36")` → séquence de zones optimale
- Pré-combat : si l'agent va affronter Brock (Roche), le graphe identifie la zone la plus proche avec des Pokémon Eau/Plante

| Tâche | Statut |
| :--- | :---: |
| Exposer `shortest_path()` dans `graph.py` | ✅ |
| Bonus navigation (+0.5) dans `_reward()` si zone sur chemin optimal | ✅ |
| Bonus type intent (+0.1/+0.2) dans `_reward()` si move SE utilisé | ✅ |
| Utiliser le chemin pour orienter les waypoints dynamiquement | ⏳ |

---

## ⏳ Phase 5 — Run complet end-to-end

> **Point d'entrée :** `run_agent.py` (existe, code complet)

| Tâche | Statut | Notes |
| :--- | :---: | :--- |
| `run_agent.py --render` fonctionne sans crash | ⏳ | Validation minimale |
| L'agent sort de la chambre seul | ⏳ | Waypoint 0 validé |
| L'agent traverse Route 1 | ⏳ | Waypoints 4-5 validés |
| L'agent traverse la Forêt de Jade | ⏳ | Waypoint 9 validé |
| L'agent gagne contre Pierre (Brock) | ⏳ | Waypoint 11 + BattleAgent validé |
| **Run complet : Bourg Palette → Badge Pierre** | ⏳ | **Objectif final du projet** |

---

## ⏳ Phase 6 — Documentation & Portfolio

| Tâche | Statut |
| :--- | :---: |
| Enregistrer une vidéo du run complet | ⏳ |
| Exporter courbes TensorBoard (reward, loss, episode length) | ⏳ |
| Mettre à jour le README avec résultats et vidéo | ⏳ |

---

## Décisions techniques

| Date | Décision |
| :--- | :--- |
| Fév 2025 | Vision/YOLO abandonné — 99% mAP obtenu sur un dataset incorrecte (modèle inutilisable). Passage à une approche RAM-only. |
| Mars 2025 | Architecture RAM-only validée : env 9-float, orchestrateur state machine, PPO curriculum two-phase. |

---

## ⏳ Phase 7 — Vision (si temps disponible)

> Option conditionnelle — uniquement si le run complet (Phase 5) est validé avant la deadline.

| Approche | Description | Statut |
| :--- | :--- | :---: |
| **YOLOv8 feature extractor** | Remplace/enrichit le vecteur RAM — CNN custom avant le PPO | ⏳ |
| **Dataset** | Régénérer depuis `debug_visualizer.py` avec les nouvelles classes | ⏳ |
| **CnnPolicy** | Passer de `MlpPolicy` à `CnnPolicy` dans SB3 | ⏳ |

> Note : L'infrastructure de base (debug_visualizer, mapping.json, ram_map section sprites) est déjà en place pour générer un dataset propre.

---

## Documents liés

| Document | Description |
| :--- | :--- |
| [ram_map.md](ram_map.md) | Adresses mémoires utilisées |
