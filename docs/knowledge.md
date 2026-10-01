# Couche de connaissance

Module `pokeblue.knowledge` (Phase 2). Il rassemble tout ce que l'agent sait du jeu
**hors-ligne** : données Gen 1 exactes, cartes, trajets, dresseurs et graphe de
progression jusqu'au Champion. La source est [pret/pokered](https://github.com/pret/pokered),
à un commit épinglé (voir `pokeblue.knowledge.pokered.source`).

```python
from pokeblue.knowledge import next_milestone, path, required_level, trainer_team, Progress

next_milestone(state)                         # jalon suivant d'après la RAM
path("PALLET_TOWN", "INDIGO_PLATEAU").maps    # trajet de cases, avec ses cartes
required_level("beat_misty")                  # 21 : Stari N21
trainer_team("BROCK", 1)                      # Racaillou N12, Onix N14 (avec Patience)
```

## Données générées

| Fichier | Générateur | Contenu |
| :--- | :--- | :--- |
| `state/ram_symbols.py` | `scripts/gen_ram_symbols.py` | adresses RAM, registres, structures |
| `knowledge/gen1_data/tables.py` | `scripts/gen_gen1_data.py` | types, attaques, espèces, évolutions, attaques apprises, dresseurs et équipes, objets, événements, charmap, palettes |
| `knowledge/data/maps/*.json` | `scripts/gen_maps.py` | 222 cartes : blocs et tuile de chaque case, connexions, warps, panneaux, objets, objets activables, rencontres sauvages, n° d'arène |
| `knowledge/data/tilesets.json` | `scripts/gen_maps.py` | tuiles praticables, contenu des blocs, règles de déplacement |

Pour tout régénérer : `pokeblue build-knowledge`. Pour vérifier que les fichiers
versionnés sont à jour : `pokeblue build-knowledge --check`, que les tests lancent aussi.

**Validation contre le jeu.** Sur les 41 savestates disponibles, les points suivants
sont identiques à la RAM :
- le tileset et les dimensions de la carte ;
- les warps (`LAST_MAP` compris) ;
- les connexions, cartes voisines et alignements compris ;
- **chaque tuile visible à l'écran**.

Les attaques calculées pour un Pokémon sauvage et pour le Pokémon du joueur
correspondent elles aussi à la RAM.

## Données relues à la main

Certaines connaissances vivent dans le code des scripts, pas dans des tables. Elles
sont décrites en YAML, chaque entrée citant son script, et testées :

- **`gates.yaml`** : 21 verrous de script de coordonnées, chacun avec sa condition
  d'ouverture.
  - Jadielle : vieil homme endormi, arène.
  - Argenta : sortie est.
  - Route 22 : porte.
  - Route 23 : les sept gardes.
  - Carmin : quai.
  - Safrania : portes.
  - Tour Pokémon : spectre.
  - Cramois'Île : arène.
  - Piste Cyclable.
  - Parc Safari : entrée.
- **`block_events.yaml`** : 68 remplacements de blocs selon un événement.
  - Route Victoire : barrières, résolubles avec Force.
  - Sylphe SARL : portes.
  - Manoir Pokémon : interrupteurs.
  - Conseil 4 : sorties.
  - Repaire Rocket : portes.
  - Casino : escalier caché.

  Les cas de forme simple sont recoupés par une extraction automatique des scripts.
  C'est cette double vérification qui a fait apparaître une condition inversée dans
  les salles du Conseil 4.
- **`progression.yaml`** : 44 jalons, du choix du starter au Champion. Chacun porte :
  - ses prérequis ;
  - sa carte cible et, le cas échéant, le dresseur à battre ;
  - sa condition d'achèvement, **vérifiable en RAM** (drapeaux, badges, objets,
    `wStatusFlags1`, objets masqués).

  Tous les verrous de l'histoire y figurent : Coupe, Flash (optionnel), Surf, Force,
  Scope Sylphe, Pokéflûte et Ronflex, Dents d'Or et Parc Safari, Carte Magnétique,
  Route Victoire.

## Navigation

`Navigator` construit un graphe de cases `(carte, x, y)` pour une `Progress` donnée.
`Progress.from_state(state)` lit les drapeaux, badges, objets et objets masqués, et
déduit les capacités : une attaque connue, plus le badge exigé hors combat. Les règles
reproduisent celles du moteur :

- **tuiles** praticables du tileset, et paires de tuiles interdites (dénivelés) ;
- **corniches** à sens unique ;
- **eau** (Surf), **arbres** (Coupe), **rochers** (Force) ;
- **warps** : une sortie `LAST_MAP` renvoie vers la carte dont un warp arrive sur
  cette porte ;
- **connexions** entre cartes, avec leur alignement ;
- **PNJ immobiles et objets activables** selon la RAM ;
- **verrous** et **remplacements de blocs** selon la progression.

`Navigator.from_state()` utilise en plus les positions courantes des PNJ : un dresseur
qui s'est avancé vers le joueur reste là où il s'est arrêté. Sur tous les savestates,
un pas dans chaque direction aboutit à la case prédite.

Résultats :
- `path(PALLET_TOWN, INDIGO_PLATEAU)` suit l'itinéraire du jeu en 0,25 s : Route 22,
  porte, Route 23, Route Victoire 1F → 2F → 3F → 2F ;
- ce trajet est impossible sans Surf, sans Badge Terre, ou sans Force tant que les
  interrupteurs ne sont pas activés ;
- les cibles des 44 jalons s'enchaînent toutes.

## Limites

- **Énigmes non modélisées** : les rochers sont tenus pour franchissables dès que
  Force est disponible. Les dalles tournantes, trous, courants et l'ascenseur du
  repaire ne sont pas modélisés : ce sont les puzzles de la Phase 5.
- **PNJ mobiles** : ceux qui se déplacent sont ignorés par la planification statique.
  `from_state` les bloque à leur position courante.
- **Combat sur l'Océane** : le combat contre le rival ne laisse pas de drapeau. Il est
  réputé fait une fois la CS01 obtenue.
- **Niveaux conseillés** : ils valent le niveau maximal de l'équipe à battre, sans
  marge. La Phase 6 les remplacera par une stratégie apprise.
- **Ancien graphe** : le graphe NetworkX alimenté par PokéAPI (`src/knowledge/`) a été
  supprimé. L'environnement RL historique utilise désormais ce module. Ses pages de
  visualisation sont archivées dans `docs/archive/`.
