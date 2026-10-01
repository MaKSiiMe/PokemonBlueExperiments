# État du jeu et détection de mode

Module `pokeblue.state` (Phase 1). Tout ce que l'agent sait du jeu passe par ici :
un instantané de la RAM, décodé en un `GameState` immuable, et un mode de jeu
déduit de l'écran.

## Lecture de la RAM

```python
from pokeblue.emulator import Emulator
from pokeblue.state.game_state import GameState
from pokeblue.state.mode_detector import detect_mode

emu = Emulator("ROMs/PokemonBlue.gb")
emu.load_state("states/37_pewter_city.state")
emu.run("wait:30 start wait:30")            # recette d'inputs
state = GameState.from_memory(emu.snapshot())
detect_mode(state)                           # Mode.MENU
```

- **`MemorySnapshot`** : WRAM (C000–DFFF) et page haute (registres IO + HRAM) lues
  en une passe. Aucune dépendance à PyBoy, d'où des tests sur mémoire synthétique.
- **`GameState`** : position, carte et orientation ; les 6 Pokémon de l'équipe (espèce,
  niveau, PV, statut, types, moves, PP hors PP Plus, stats, expérience) ; sac, argent
  (BCD) et badges ; les **2 560 event flags** en bitset (`state.flag(EVENT_IDS[...])`) ;
  Pokédex ; structures de combat `wBattleMon` / `wEnemyMon` avec paliers de stats ;
  tilemap et registres utilisés par la détection de mode.
- **`GameState.diff(prev)`** : drapeaux activés, changement de carte, badges, K.O.,
  montées de niveau, début/fin de combat, K.O. ennemi, argent, objets, Pokédex.
  `describe()` donne des lignes lisibles pour les logs.

Toutes les adresses et tailles viennent de `ram_symbols.py`, généré depuis
`pokeblue.sym` (voir `scripts/gen_ram_symbols.py`). Un test échoue si une adresse
RAM est écrite en dur ailleurs.

Coût mesuré (Ryzen 5 5600H) : instantané 203 µs, `GameState` 18 µs, `detect_mode`
9 µs, pour une action de 24 frames à 1,33 ms (+17 %). La copie de la WRAM domine :
ne lire que les plages utiles est la première optimisation si le débit compte.

## Modes

| Mode | Signification | Règle |
| :--- | :--- | :--- |
| `TRANSITION` | fondu, chargement de carte, entrée en combat | LCD éteint, ou `rBGP` ≠ palette stable de la carte, ou combat sans boîte de texte |
| `BATTLE_MOVE_MENU` | choix de l'attaque | en combat, curseur ▶ et cadre « TYPE/ » |
| `BATTLE_MENU` | menu de combat, sac, équipe, OUI/NON | en combat, curseur ▶ |
| `BATTLE_ANIM` | messages et animations de combat | en combat, boîte de texte sans curseur |
| `MENU` | menu Start, sac, équipe, Pokédex, boutique, PC, OUI/NON | curseur ▶ dans le tilemap |
| `DIALOG` | dialogues, panneaux, écrans d'information | tuiles d'interface (cadres, police) sans curseur |
| `OVERWORLD` | déplacement libre | aucune tuile d'interface |

Il n'existe pas d'octet « dialogue actif » en Gen 1. Les règles reposent sur :

- **le tilemap** (`wTileMap`, 20 × 18 tuiles). Dans l'overworld, il ne contient que
  des tuiles de décor (jeu de tuiles chargé à partir de la tuile 0). Les cadres
  (`┌─┐│└┘`), la police et le curseur ▶ ont des tuiles dédiées (`charmap.asm`) ;
- **la palette de fond** `rBGP`. Les fondus passent par `FadePal1..8`
  (`home/fade.asm`). La palette stable est `FadePal4` décalée de `wMapPalOffset`,
  donc la détection reste correcte dans une grotte sombre ;
- **`wIsInBattle`**, positionné dès la spirale d'entrée en combat.

Chronologies relevées (en frames), qui ont servi à établir ces règles :

- **Porte** :
  - frame 18 : `wCurMap` passe à la destination ;
  - frames 26–62 : `rBGP` passe de `e4` à `f9`, `fe` puis `ff` (fondu au noir) ;
  - frames 50–55 : LCD éteint, `wJoyIgnore = ff` ;
  - frame 63 : retour à `e4`.
- **Combat sauvage** :
  - frames 0–61 : spirale (`wIsInBattle = 1`, pas de boîte de texte) ;
  - frames 62–125 : écran noir (`rBGP = ff`) ;
  - frame 342 : texte d'intro complet ;
  - après A : « Go! CHARMANDER! » ;
  - frame 588 : menu FIGHT, qui s'affiche seul.

## Jeu de savestates étiquetés

`configs/states/modes.yaml` décrit 33 états couvrant les 7 modes. Chacun est défini
par un savestate de base de `states/` et une recette d'inputs, ce qui le rend
régénérable à l'identique. Les étiquettes ont été attribuées d'après la capture
d'écran de chaque état, indépendamment du détecteur.

```bash
pokeblue make-states            # states/modes/*.state, *.png et sheet.png
pytest tests/test_mode_detector.py
```

Résultat : **33/33** (`tests/test_mode_detector.py`). Pour ajouter un cas : ajouter
une entrée au manifeste, régénérer, vérifier la capture sur `sheet.png`, puis lancer
les tests.

Syntaxe des recettes (`pokeblue.emulator.parse_recipe`) : `a`, `b`, `start`, `select`,
`up`/`down`/`left`/`right` (une action de 24 frames), `up*3`, `wait:60`,
`hold:left:6` (bouton maintenu 6 frames sans compléter l'action).

## Durées d'appui (mesurées)

- Dans l'overworld, le jeu ne lit la manette qu'une frame sur deux : un appui d'une
  frame est perdu une fois sur deux (12/24 décalages), jamais à partir de 2 frames.
  Les boutons sont donc pressés `PRESS_FRAMES = 4` frames.
- Une direction maintenue de 2 à 17 frames fait exactement un pas ; au-delà, le jeu
  enchaîne un second pas (23 frames = 2 cases). Les directions sont maintenues
  `HOLD_FRAMES = 8` frames.

L'environnement RL historique pressait une frame et maintenait 23 frames. Il a été
corrigé, mais les checkpoints antérieurs ont appris avec ces actions faussées.

## Outil de debug

```bash
pokeblue overlay --state states/37_pewter_city.state            # fenêtre (extra tools)
pokeblue overlay --state ... --recipe "wait:30 start" --out p.png
```

Le panneau affiche le mode détecté, la carte, l'équipe, le sac, les deux Pokémon en
combat, le texte à l'écran et le journal des `StateDiff`.

## Limites connues

- **Blip de transition** : quand A fait avancer un texte de combat, l'écran est vidé
  pendant une frame, qui est classée `TRANSITION`. C'est sans conséquence pour un
  orchestrateur qui attend.
- **Début de porte** : `wCurMap` change environ 8 frames avant le début du fondu.
  Ces frames sont classées `OVERWORLD`.
- **Pas de distinction texte/animation** : en combat, `BATTLE_ANIM` ne distingue pas
  un texte qui attend A d'une animation. Appuyer sur A est sans risque dans les deux
  cas. L'invite ▼ clignote et n'est donc pas un signal fiable.
- **Écrans non couverts** par le jeu étiqueté : titre et nouvelle partie, saisie de
  nom, évolution, page d'une entrée du Pokédex, échange, Zone Safari, pêche,
  sous-menus du PC. À ajouter au manifeste quand un savestate de base existera.
- **Savestates de base** : ceux de `states/` sont des sauvegardes manuelles. Un
  savestate régénérable depuis l'allumage de la console reste à écrire. Au passage,
  `48_pewter_gym_badge.state` précède le combat contre Pierre : le badge n'y est pas.
