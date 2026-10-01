# Baseline scriptée (Phase 3)

Un agent entièrement scripté qui joue depuis la nouvelle partie, sans apprentissage.
Il sert de référence pour les modules appris des phases suivantes : chaque skill appris
devra faire mieux que sa version scriptée, mesurée par le même harnais (`pokeblue eval`).

## Architecture

```
GameState (RAM) ──► Orchestrateur ──► Skill actif ──► bouton (action de 24 frames)
                        ▲   │
                Stratégie   └─ interruptions : combat → BattleSkill,
               (objectif)      dialogue/menu → politique de dialogue,
                               capacité de terrain demandée → FieldMoveSkill
```

### Contrat des skills — `pokeblue/skills/base.py`

Un skill reçoit un `Goal(kind, params)`, puis à chaque pas le `GameState`, et renvoie un
bouton (ou `None` pour laisser le jeu avancer). `status()` vaut `RUNNING`, `SUCCESS`,
`FAILURE` (motif dans `failure`) ou `TIMEOUT` (au-delà de `budget_steps` actions).
Une version apprise d'un skill garde ce contrat : l'orchestrateur ne change pas.

| skill | fichier | rôle |
|---|---|---|
| `NavigationSkill` | `skills/navigation.py` | plus court chemin sur les grilles de la Phase 2 (warps, connexions, corniches, PNJ en direct), replanification, contournement d'un PNJ qui bloque, sortie d'un warp sur lequel on vient d'arriver, aborder un objet (y compris par-dessus un comptoir), demande de Coupe/Surf |
| `BattleSkill` | `skills/battle.py` | attaque = multiplicateur de type Gen 1 × puissance × STAB × précision × rapport de stats, PP et Entrave pris en compte ; Potion sous un seuil ; fuite des combats sauvages inutiles ou mal engagés ; nouvelle attaque apprise si elle bat la plus faible |
| `DialogSkill` / `dialog_button` | `skills/dialog.py` | A sur les textes ; OUI sauf surnom ; SOIGNER chez l'infirmière ; B sur un menu inattendu |
| `HealSkill`, `TrainSkill` | `skills/field.py` | Centre Pokémon le plus proche ; allers-retours dans l'herbe jusqu'au niveau visé |
| `TeachMoveSkill`, `FieldMoveSkill`, `BuySkill` | `skills/menus.py` | apprendre une CS (oubli de l'attaque la moins utile), utiliser Coupe depuis le menu Équipe, acheter des potions |
| `TrashCanSkill` | `skills/puzzles.py` | énigme des poubelles de Carmin, résolue « en joueur informé » (table `GymTrashCans` de pokered, jamais les index cachés en RAM) |

Les menus sont lus à l'écran (`skills/menu_reader.py` : curseur ▶, options alignées,
texte décodé par la charmap pokered), jamais pilotés par des séquences aveugles.

### Stratégie — `pokeblue/orchestrator/strategy.py`

Règles, dans l'ordre, réévaluées à chaque point de décision :

1. équipe vide → jalon courant (le starter) ;
2. PV de l'équipe sous 50 %, Pokémon K.O. ou plus de PP offensifs → Centre Pokémon ;
3. CS dans le sac + badge → l'apprendre ; moins de 3 potions et ≥ 1500 ¥ → boutique la
   plus proche qui en vend (inventaires extraits de `data/items/marts.asm`) ;
4. niveau maximal sous le niveau conseillé du jalon (`required_level`) → entraînement
   sur l'herbe atteignable la plus proche dont les sauvages ne sont pas trop forts ;
5. sinon → objectif du jalon : champ `approach` / `steps` de `progression.yaml` (case
   de déclenchement d'un script relevée dans `scripts/*.asm`, objet à aborder, skill
   dédié), ou par défaut parler au dresseur ciblé, ou entrer dans la carte cible.

Le choix du starter (Bulbizarre, efficace contre Pierre et Ondine et capable d'apprendre
Coupe) est une décision de la baseline, écrite dans `progression.yaml`.

### Orchestrateur — `pokeblue/orchestrator/core.py`

Ordre de priorité à chaque pas : combat → transition (on attend) → capacité de terrain
demandée → décision (si point de décision) → dialogue non géré par le skill → skill
actif. Tout `FAILURE`/`TIMEOUT` est journalisé avec un savestate. Après 3 échecs
consécutifs du même objectif, une marche aléatoire de 24 pas est intercalée ; après 12
échecs consécutifs, ou 20 000 actions sans nouveau jalon, le run s'arrête (`stuck`,
`stalled`) : l'orchestrateur ne reste jamais bloqué indéfiniment.

## Évaluation

```bash
pokeblue eval --runs 8 --workers 8          # configs/eval/baseline.yaml
pokeblue run --start-state <savestate d'échec>   # rejouer un échec
```

Chaque run part de `states/PokemonBlue.gb.state` (nouvelle partie à Bourg Palette) et
attend 0 à 600 frames tirées de sa graine : le générateur aléatoire du jeu avance à
chaque frame, donc rencontres, dégâts et coups critiques diffèrent d'un run à l'autre.
Une action dure 24 frames (0,4 s de jeu).

### Résultats (2026-10-02, 8 runs, `logs/eval/phase3-final`)

8 parties sur 8 obtiennent **4 badges** (Roche, Cascade, Foudre, Prisme) depuis la
nouvelle partie, sans intervention. Les 8 runs s'arrêtent au même jalon (`get_drink`).
Durée totale : 194 s de calcul pour 8 runs en parallèle, soit environ 1 500 actions
par seconde et par instance.

| jalon | runs | actions : médiane (min–max) |
|---|---|---|
| get_pokedex | 100 % | 965 (870–1080) |
| beat_brock — Badge Roche | 100 % | 8 000 (7 120–8 880) |
| cross_mt_moon | 100 % | 13 325 (11 120–14 310) |
| beat_misty — Badge Cascade | 100 % | 15 430 (12 530–18 640) |
| get_ss_ticket | 100 % | 18 850 (15 480–22 620) |
| get_hm01_cut | 100 % | 25 175 (20 970–27 770) |
| beat_lt_surge — Badge Foudre | 100 % | 27 410 (22 240–30 100) |
| pokemon_tower_rival | 100 % | 31 615 (27 560–36 130) |
| beat_erika — Badge Prisme | 100 % | 33 605 (29 000–39 110) |

Écart entre badges (médianes) : 8 000 actions jusqu'au 1er, puis 7 400, 12 000 et 6 200.
Le premier écart comprend l'entraînement de Bulbizarre du niveau 5 au niveau 14.
Pokémon de l'équipe tombés K.O. : médiane 7 (3–9) par run. Comme l'équipe ne compte
qu'un Pokémon, chaque K.O. est un blackout et un retour au dernier Centre.

### Où la baseline casse

- **Distributeur du toit de Céladopole (`get_drink`)** : acheter une boisson passe par
  un menu propre au distributeur, non scripté. Les 8 runs s'arrêtent là
  (`stalled` : 20 000 actions sans nouveau jalon). La navigation atteint le toit, puis
  boucle sur « objectif atteint sans progrès » (≈ 2 200 échecs journalisés par run).
- **Équipe d'un seul Pokémon** : aucune capture. Le rival de l'Océane (Reptincel) et
  les dresseurs à attaques de statut (Paralysie, Entrave) provoquent l'essentiel des
  K.O. Les potions achetées et la fuite des combats sauvages les limitent, sans
  les supprimer.
- **Couloirs étroits avec PNJ mobiles** (Forêt de Jade, case (1,18)) : « aucun chemin »
  tant que le PNJ bloque. Rattrapé par la patience de replanification, puis par la
  marche aléatoire.
- **Transitions de carte** (Route 3 → Mont Sélénite) : un échec « aucun chemin » isolé
  par run, sans conséquence.
- **Simplifications connues** :
  - les CS sont apprises au 1er Pokémon, car les compatibilités CT/CS ne sont pas
    extraites ;
  - l'achat se fait à l'unité ;
  - Surf et Force ne sont pas encore utilisés hors combat (aucun jalon atteint ne les
    exige) ;
  - les énigmes de rochers, dalles tournantes et Flash ne sont pas traités.

### Bugs de données révélés par la baseline (corrigés)

- `UNDERGROUND_PATH_ROUTE_7` recevait les warps de son double inutilisé
  `UndergroundPathRoute7Copy` : le générateur prend maintenant le header que
  `MapHeaderPointers` associe à chaque carte (test de non-régression ajouté).
- Après une défaite, le rival de l'Océane reste affiché devant la porte du capitaine :
  la porte était vue comme bloquée. Une étape `steps` renvoie sur la case qui relance
  le combat.
