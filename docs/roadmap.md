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

### Curriculum — 13 waypoints (Bourg Palette → Badge Pierre)

| # | Label | Map cible | State de départ | Budget/ep |
| :---: | :--- | :---: | :--- | :---: |
| 0 | Chambre → escalier | `0x26` | `states/01_chambre.state` | 30 |
| 1 | Maison 1F → porte sud | `0x25` | `states/02_maison_1f.state` | 50 |
| 2 | Entrer au Labo Chen | `0x00` | `states/06_pallet_town.state` | 200 |
| 3 | Prendre la Pokéball | `0x52` | `states/04_lab.state` | 150 |
| 4 | Sortir du Labo | `0x00` | `states/04_lab.state` | 150 |
| 5 | Route 1 — vers nord | `0x12` | `states/06_pallet_town.state` | 400 |
| 6 | Arriver à Jadielle City | `0x01` | `states/07_route1_grass.state` | 600 |
| 7 | Entrer au Poké Mart | `0x01` | `states/11_viridian_center.state` | 400 |
| 8 | Retour Labo Chen | `0x00` | `states/11_viridian_center.state` | 800 |
| 9 | Route 2 | `0x13` | `states/22_route2_down.state` | 800 |
| 10 | Forêt de Jade | `0x33` | `states/24_viridian_forest_down.state` | 1200 |
| 11 | Argenta City | `0x02` | `states/35_pewter_center_front.state` | 1200 |
| 12 | Arène Argenta → Brock | `0x54` | `states/46_pewter_gym_front.state` | 800 |

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

## Documents liés

| Document | Description |
| :--- | :--- |
| [ram_map.md](ram_map.md) | Adresses mémoires utilisées |
