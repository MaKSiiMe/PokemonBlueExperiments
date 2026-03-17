# 🧠 RAM Map : Pokémon Blue (US/EU)

Référence unique des adresses mémoires utilisées via PyBoy pour piloter les agents IA (observations, reward, orchestration).

---

## 🧭 1. Orchestrateur (Contexte Global)

Ces adresses permettent a l'orchestrateur de savoir quel module (Combat, Nav) doit prendre le controle.

| Variable | Adresse | Type | Description |
| :--- | :--- | :--- | :--- |
| **Battle Status** | `0xD057` | `uint8` | `00` overworld, `!= 00` en combat. |
| **Battle Type** | `0xD05A` | `uint8` | Type de combat (sauvage, dresseur, etc.). |
| **Menu ID** | `0xD12B` | `uint8` | `00` aucun menu, >`0` menu/sac/stats ouvert. |
| **Text ID** | `0xD11C` | `uint8` | >`0` boite de dialogue active. |
| **Fading State** | `0xD13F` | `uint8` | `00` pret, `01-FF` transition (ecran noir). |
| **In-Game Timer** | `0xDA43` | `uint8` | Secondes de jeu (utile pour horodater le dataset). |

---

## 👁️ 2. Sprites & Tiles (Debug Visualizer)

Utilisé par `src/utils/debug_visualizer.py` pour dessiner les entités en overlay.

### Classes d'entités

| ID | Classe | Source | Description |
| :---: | :--- | :--- | :--- |
| `0` | **Player** | Sprite WRAM | Toujours slot 0 à `0xC100`, toujours en case (4,4) = pixel (72,72) |
| `1` | **NPC** | Sprite WRAM | Tous les sprites actifs sauf Player et Item |
| `2` | **Door** | Tile ID | Portes, escaliers, tapis de sortie — transitions de zone |
| `3` | **Tall Grass** | Tile ID | Hautes herbes — déclenchent des combats aléatoires |
| `4` | **Ledge** | Tile ID | Corniches directionnelles — passage dans un seul sens |
| `5` | **Item** | Sprite WRAM | Pokéballs et objets au sol (sprite `0x3C`) |

> IDs de tiles à confirmer zone par zone avec `debug_visualizer.py`. Voir `mapping.json` section `tiles`.

### A. Objets Dynamiques (Sprites)

Utiliser exclusivement la table WRAM `0xC100` — elle contient l'ID logique du sprite (ex: `0x01` = joueur, `0x3C` = item).
OAM `0xFE00` ne contient pas d'ID logique — utile uniquement pour filtrer les sprites fantômes.

| Source | Adresse | Description |
| :--- | :--- | :--- |
| **WRAM Sprites** | `0xC100 - 0xC2FF` | Table logique (16 octets/sprite). Source principale pour l'identification. |
| **Hardware OAM** | `0xFE00 - 0xFE9F` | Table materielle (4 octets/sprite). Precision pixel — filtre les sprites fantômes. |

Offsets critiques — table principale WRAM `0xC100` (16 slots × 16 octets) :

| Offset | Champ | Description |
| :---: | :--- | :--- |
| `+0x00` | **Picture ID** | Type du sprite — clé dans `mapping.json["sprites"]` (`0` = slot inactif). **C'est ici le vrai identifiant du personnage.** |
| `+0x01` | **Movement Status** | `0`=non init, `1`=prêt, `2`=délai, `3`=en mouvement. |
| `+0x02` | **Sprite Image Index** | ⚠️ Index d'animation uniquement (direction + frame). Pas le type de sprite. |
| `+0x04` | **Y Screen** | Position Y écran (pixels bruts). |
| `+0x06` | **X Screen** | Position X écran (pixels bruts). |
| `+0x09` | **Facing Direction** | `0x00`=bas, `0x04`=haut, `0x08`=gauche, `0x0C`=droite. |

> Slot 0 (`0xC100`) = toujours le joueur.

Offsets critiques — table étendue `0xC200` (16 slots × 16 octets) :

| Offset | Champ | Description |
| :---: | :--- | :--- |
| `+0x04` | **Y Block** | Position Y en blocs (unités de 2 cases). |
| `+0x05` | **X Block** | Position X en blocs (unités de 2 cases). |
| `+0x07` | **Grass Flag** | `0x80` si le sprite est dans les hautes herbes, `0x00` sinon. |

> Pour le joueur : `0xC207` = grass flag. Utile pour détecter l'entrée en hautes herbes **sans YOLO**.

### B. Objets Statiques (Tiles)

Les tuiles de fond sont lues depuis le buffer écran — grille **20 colonnes × 18 lignes** de tuiles **8×8px**.
Les objets logiques (portes, herbes, corniches) font **16×16px** = 2×2 tuiles hardware.

| Variable | Adresse | Description |
| :--- | :--- | :--- |
| **Tile Map Buffer** | `0xC3A0 - 0xC507` | 360 octets (20×18 tiles de 8×8px). |
| **Previous Tile Buffer** | `0xC508 - 0xC5CF` | Buffer précédent (restauration après menu). |
| **Current Tileset** | `0xD367` | ID du pack graphique actif — conditionne les tile IDs. |
| **Map Height** | `0xD368` | Hauteur de la map en blocs. |
| **Map Width** | `0xD369` | Largeur de la map en blocs. |
| **SCX (hardware)** | `0xFF43` | Scroll X en pixels — registre hardware Game Boy. |
| **SCY (hardware)** | `0xFF42` | Scroll Y en pixels — registre hardware Game Boy. |
| **SCX (WRAM copy)** | `0xFFAE` | Copie WRAM du scroll X (source : Data Crystal). |
| **SCY (WRAM copy)** | `0xFFAF` | Copie WRAM du scroll Y (source : Data Crystal). |

> ⚠️ Le même objet visuel (ex: porte) a des tile IDs différents selon le tileset. Toujours lire `0xD367` avant de scanner les tiles.

### C. Grille écran

L'écran Game Boy est un damier de **10 colonnes × 9 lignes** de cases logiques **16×16px**.
Le joueur est toujours en case **(col 4, row 4)** → centre pixel **(72, 72)**.

```
Grille logique 10×9 (cases 16px) :
col  0    1    2    3    4    5    6    7    8    9
    [  ] [  ] [  ] [  ] [PL] [  ] [  ] [  ] [  ] [  ]  row 4
px   8   24   40   56   72   88  104  120  136  152
```

---

## 📍 3. Navigation (Overworld)

Donnees pour l'IA de pathfinding (A*).

| Variable | Adresse | Description |
| :--- | :--- | :--- |
| **Y Pos (cases)** | `0xD361` | Position Y en cases (`0-255`). |
| **X Pos (cases)** | `0xD362` | Position X en cases (`0-255`). |
| **Y Pos (blocs)** | `0xD363` | Position Y en blocs (1 bloc = 2 cases). |
| **X Pos (blocs)** | `0xD364` | Position X en blocs (1 bloc = 2 cases). |
| **Direction** | `0xD35D` | `00` bas, `04` haut, `08` gauche, `0C` droite. |
| **Map ID** | `0xD35E` | ID de la zone (ex: `02` pour Argenta). |
| **Map Connection** | `0xD35F` | `uint8` | Permet de savoir si on change de map. |
| **Map Tileset** | `0xD367` | ID du tileset graphique actif. |
| **Map Height** | `0xD368` | Hauteur de la map en blocs. |
| **Map Width** | `0xD369` | Largeur de la map en blocs. |

Points de passage cles (Map IDs) — Source : `constants/map_constants.asm` (pret/pokered) :

| Map ID (hex) | Nom FR | Nom EN | Dimensions |
| :---: | :--- | :--- | :--- |
| `0x00` | Bourg Palette | Pallet Town | 10×9 |
| `0x01` | Jadielle | Viridian City | 20×18 |
| `0x02` | Argenta (Pierre) | Pewter City | 20×18 |
| `0x0C` | Route 1 | Route 1 | 10×18 |
| `0x0D` | Route 2 | Route 2 | 10×? |
| `0x17` | Route 22 | Route 22 | ?×? |
| `0x33` | Forêt de Jade | Viridian Forest | ?×? |

---

## ⚔️ 4. Combat (Battle Engine)

Donnees pour l'IA de decision tactique.

| Variable | Adresse | Type | Description |
| :--- | :--- | :--- | :--- |
| **Enemy HP** | `0xCFE6` | `uint16` | PV actuels de l'adversaire (Little Endian). |
| **Player HP** | `0xD16C` | `uint16` | PV actuels de ton Pokemon actif. |
| **Player Max HP** | `0xD18D` | `uint16` | PV max. |
| **Move 1..4** | `0xD173 - 0xD176` | `uint8` | IDs des attaques disponibles. |
| **PP 1..4** | `0xD188 - 0xD18B` | `uint8` | Points de pouvoir restants. |
| **Enemy ID** | `0xD011` | `uint8` | ID du Pokemon adverse. |
| **Enemy Move** | `0xD01C` | `uint8` | ID de l'attaque que l'ennemi va lancer. |
| **Enemy Max HP** | `0xCFE8` | `uint16` | PV max. |
| **Enemy Level** | `0xD012` | `uint8` | Niveau de l'adversaire. |

---

## 🎒 5. Progression & Inventaire

| Variable | Adresse | Description |
| :--- | :--- | :--- |
| **Money** | `0xD347` | Argent (3 octets, format BCD). |
| **Badges** | `0xD356` | Bitmask des badges (Pierre = 1er bit). |
| **Pokedex** | `0xD2F7` | Nombre de Pokemon possedes. |

---

## 🧪 6. Implementation Python (Cheat Sheet)

Lecture securisee des PV (16-bit Big Endian — convention Gen 1)

```python
def read_u16(pyboy, addr):
    return (pyboy.memory[addr] << 8) | pyboy.memory[addr + 1]

player_hp     = read_u16(pyboy, 0xD16C)
player_max_hp = read_u16(pyboy, 0xD18C)
hp_pct        = player_hp / max(player_max_hp, 1)
```

Lecture position joueur

```python
x      = pyboy.memory[0xD362]
y      = pyboy.memory[0xD361]
map_id = pyboy.memory[0xD35E]
```
