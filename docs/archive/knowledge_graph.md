# Knowledge Graph — Comment l'IA lit sa carte stratégique

> Ce document explique la structure du graphe de connaissances Gen 1, comment le lire,
> et comment l'IA l'utilise concrètement pour prendre des décisions.

---

## 1. L'unité de base : le Triplet

Toute l'intelligence du graphe repose sur une structure en trois parties appelée **triplet**.
Ce n'est pas une ligne de données — c'est une **affirmation**.

```
(Sujet) --[Prédicat]--> (Objet)
```

| Partie | Rôle | Exemple |
| :--- | :--- | :--- |
| **Sujet** | L'entité de départ | `pokemon:squirtle` |
| **Prédicat** | La nature de la relation | `HAS_TYPE` |
| **Objet** | L'entité d'arrivée | `type:water` |

```
(squirtle) --[HAS_TYPE]--> (water)
```

> Cette affirmation apprend à l'IA une propriété intrinsèque de l'entité.

---

## 2. Les nœuds du graphe

Le graphe contient **4 types de nœuds** (337 au total) :

| Type | Préfixe | Exemple | Attributs stockés |
| :--- | :--- | :--- | :--- |
| `pokemon` | `pokemon:` | `pokemon:pikachu` | dex, types, hp, attack, defense, speed |
| `type` | `type:` | `type:electric` | name |
| `move` | `move:` | `move:thunderbolt` | move_id, type_name, base_power, damage_class, priority, pp |
| `zone` | `zone:` | `zone:0x33` | name, map_id, pokeapi_slug |

---

## 3. Les relations (arêtes)

Le graphe est un **DiGraph** (graphe orienté) : **le sens de la flèche est capital**.

| Relation | Direction | Lecture | Utilité IA |
| :--- | :--- | :--- | :--- |
| `HAS_TYPE` | pokemon → type | "A pour type B" | Connaître forces/faiblesses |
| `LEARNS` | pokemon → move | "Peut apprendre l'attaque B" | Anticiper les capacités adverses |
| `IS_TYPE` | move → type | "L'attaque B est de type C" | Calculer le multiplicateur |
| `SUPER_EFFECTIVE_AGAINST` | type → type | "Le type A bat le type B" (×2.0) | Choisir la meilleure attaque |
| `NOT_EFFECTIVE_AGAINST` | type → type | "Le type A est résisté par B" (×0.5) | Éviter les attaques inutiles |
| `NO_EFFECT_AGAINST` | type → type | "Le type A est immunisé par B" (×0.0) | Ne jamais utiliser cette combo |
| `FOUND_IN` | pokemon → zone | "Se rencontre dans la zone B" | Savoir où aller pour capturer |
| `LEADS_TO` | zone → zone | "La zone A mène à la zone B" | Pathfinding (navigation) |
| `EVOLVES_INTO` | pokemon → pokemon | "A évolue en B" | Prédire le danger futur |

### Chiffres actuels

```
337 nœuds  : 151 pokemon · 163 move · 15 type · 8 zone
4 397 arêtes : 3850 LEARNS · 211 HAS_TYPE · 162 IS_TYPE · 72 EVOLVES_INTO
               39 SUPER_EFFECTIVE · 38 NOT_EFFECTIVE · 6 NO_EFFECT
               12 FOUND_IN · 7 LEADS_TO
```

---

## 4. Lire le graphe dans deux directions

Parce que le graphe est orienté, chaque nœud a des **arêtes sortantes** et des **arêtes entrantes**.

### Lecture directe (sortante)

```python
# "Pikachu évolue en quoi ?"
G.out_edges("pokemon:pikachu", data=True)
# → ("pokemon:pikachu", "pokemon:raichu", {"relation": "EVOLVES_INTO"})
```

### Lecture inverse (entrante)

```python
# "Qui évolue en Raichu ?"
G.in_edges("pokemon:raichu", data=True)
# → ("pokemon:pikachu", "pokemon:raichu", {"relation": "EVOLVES_INTO"})
```

---

## 5. Le Multi-hop : suivre le fil d'Ariane

C'est là que l'IA devient réellement intelligente.
Un **multi-hop** consiste à enchaîner plusieurs triplets pour déduire une conclusion
**qui n'est pas écrite explicitement** dans le graphe.

### Exemple : l'IA choisit la meilleure attaque

L'agent est face à un Racaillou. Il veut savoir si "Pistolet à O" est efficace.

```
Saut 1 : (move:water-gun)  --[IS_TYPE]-->               (type:water)
Saut 2 : (type:water)      --[SUPER_EFFECTIVE_AGAINST]-> (type:rock)
Saut 3 : (pokemon:geodude) --[HAS_TYPE]-->               (type:rock)

Conclusion : "Pistolet à O" × 2.0 contre Racaillou.
```

En Python, cette déduction est encapsulée dans `graph.py` :

```python
kg.type_multiplier("water", ["rock"])   # → 2.0
```

### Exemple : l'IA anticipe l'évolution adverse

```
Saut 1 : (pokemon:caterpie) --[EVOLVES_INTO]--> (pokemon:metapod)
Saut 2 : (pokemon:metapod)  --[EVOLVES_INTO]--> (pokemon:butterfree)
Saut 3 : (pokemon:butterfree) --[HAS_TYPE]-->   (type:bug)
Saut 4 : (pokemon:butterfree) --[HAS_TYPE]-->   (type:flying)

Conclusion : Si l'IA voit un Chenipan, elle sait qu'il deviendra
             Papilusion (Bug/Vol) — elle doit préparer des attaques Feu ou Roche.
```

```python
kg.evolution_chain("caterpie")          # → ["caterpie", "metapod", "butterfree"]
kg.pokemon_types("butterfree")          # → ["bug", "flying"]
```

### Exemple : l'IA planifie son chemin

```
Saut 1 : (zone:0x00) --[LEADS_TO]--> (zone:0x0c)   # Bourg Palette → Route 1
Saut 2 : (zone:0x0c) --[LEADS_TO]--> (zone:0x01)   # Route 1 → Jadielle
Saut 3 : (zone:0x01) --[LEADS_TO]--> (zone:0x0d)   # Jadielle → Route 2
...

Conclusion : Chemin optimal jusqu'à l'Arène = 6 sauts.
```

```python
nx.shortest_path(G, "zone:0x00", "zone:0x36")
# → ["zone:0x00", "zone:0x0c", "zone:0x01", "zone:0x0d", "zone:0x0e", "zone:0x33", "zone:0x02", "zone:0x36"]
```

---

## 6. Quirks Gen 1 encodés dans le graphe

Le graphe utilise les règles **Gen 1 strictes**, pas les règles modernes.
Ces différences sont critiques pour ne pas prendre de mauvaises décisions tactiques.

| Relation | Gen 1 | Gen 2+ | Impact |
| :--- | :---: | :---: | :--- |
| Ghost → Psychic | **0.0** (bug) | 2.0 | Ne jamais utiliser Léchouille vs Alakazam |
| Poison → Bug | **2.0** | 1.0 | Dard-Venin est SE contre Chenipan |
| Bug → Psychic | **2.0** | 0.5 | Jackpot Bug contre Alakazam |
| Bug → Poison | **2.0** | 1.0 | Jackpot Bug contre Nidoran |
| Ice → Fire | **2.0** | 1.0 | Blizzard est SE contre Caninos |

---

## 7. Comment l'IA utilise le graphe concrètement

### Dans `BattleAgent` (combat heuristique)

```python
# Lit les types ennemis depuis la RAM
enemy_type_bytes = [pyboy.memory[0xD01F], pyboy.memory[0xD020]]

# Calcule le multiplicateur pour chaque move via le graphe
score = kg.type_multiplier_from_ram(move_type_byte, enemy_type_bytes)
# → 0.0 (immunisé) / 0.5 (résisté) / 1.0 (neutre) / 2.0 (SE) / 4.0 (double SE)
```

**Avant le graphe** : le scoring était binaire (1 ou 2). Les immunités et résistances étaient ignorées.
**Après le graphe** : 5 niveaux de multiplicateur, Ghost→Psychic correct, résistances évitées.

### Dans `PokemonBlueEnv` (action masking)

```python
def action_masks(self) -> np.ndarray:
    # En combat : masque les actions directionnelles
    # Si tous les moves sont immunisés (0.0x) → masque aussi 'a'
    mult = TYPE_CHART.get(move_type).get(enemy_type, 1.0)
    if mult == 0.0:
        mask[4] = False  # 'a' désactivé
```

**Résultat** : le réseau de neurones PPO n'explore jamais des actions inutiles.
L'espace de recherche effectif est réduit → convergence plus rapide.

---

## 8. Requêtes disponibles dans `graph.py`

| Méthode | Entrée | Sortie | Usage |
| :--- | :--- | :--- | :--- |
| `type_multiplier(atk, defs)` | noms de types | float | Score d'attaque |
| `type_multiplier_from_ram(byte, bytes)` | octets RAM | float | BattleAgent direct |
| `best_move_index(names, bytes, hp%)` | slots + RAM | int 0-3 | Sélection move |
| `encounters_in_zone(map_id)` | int | list[str] | Prédiction rencontres |
| `next_zones(map_id)` | int | list[int] | Navigation |
| `zone_type_threat(map_id, types)` | int + types | float | Danger de la zone |
| `evolutions(name)` | str | list[str] | Évolutions directes |
| `evolution_chain(name)` | str | list[str] | Chaîne complète |
| `pokemon_types(name)` | str | list[str] | Types d'un Pokémon |
| `pokemon_stats(name)` | str | dict | Stats de base |

---

## Documents liés

| Document | Description |
| :--- | :--- |
| [ram_map.md](ram_map.md) | Adresses mémoire utilisées pour lire les types/moves depuis la RAM |
| [roadmap.md](roadmap.md) | Prochaines étapes : augmentation de l'observation (Vecteur KG) et pathfinding |
