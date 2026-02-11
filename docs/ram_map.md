# 🧠 RAM Map : Pokémon Blue (US/EU)

Ce document est la reference unique pour l'extraction de donnees via PyBoy. Il permet d'automatiser le dataset YOLO et de piloter les sous-modules de l'IA.

---

## 🧭 1. Orchestrateur (Contexte Global)

Ces adresses permettent a l'orchestrateur de savoir quel module (Vision, Combat, Nav) doit prendre le controle.

| Variable | Adresse | Type | Description |
| :--- | :--- | :--- | :--- |
| **Battle Status** | `0xD057` | `uint8` | `00` overworld, `01` sauvage, `02` dresseur. |
| **Menu ID** | `0xD12B` | `uint8` | `00` aucun menu, >`0` menu/sac/stats ouvert. |
| **Text ID** | `0xD11C` | `uint8` | >`0` boite de dialogue active. |
| **Fading State** | `0xD13F` | `uint8` | `00` pret, `01-FF` transition (ecran noir). |
| **In-Game Timer** | `0xDA43` | `uint8` | Secondes de jeu (utile pour horodater le dataset). |

---

## 👁️ 2. Vision (Dataset YOLO & Calibration)

Utilise pour generer les labels .txt automatiquement pendant tes 2h de jeu.

### A. Objets Dynamiques (Sprites)

Il existe deux manieres de voir les objets. La table C100 est preferable pour YOLO car elle contient l'ID logique de l'objet (ex: "Scientifique").

| Source | Adresse | Description |
| :--- | :--- | :--- |
| **WRAM Sprites** | `0xC100 - 0xC2FF` | Table logique (16 octets/sprite). Recommande pour YOLO. |
| **Hardware OAM** | `0xFE00 - 0xFE9F` | Table materielle (4 octets/sprite). Utile pour la precision pixel. |

Offsets critiques (WRAM `0xC100`) :

| Offset | Champ | Description |
| :---: | :--- | :--- |
| `+0x00` | **Picture ID** | Indique si le sprite est actif (`0` = inactif). |
| `+0x01` | **Movement Status** | Indique si le sprite bouge. |
| `+0x02` | **Sprite ID** | Classe YOLO. |
| `+0x04` | **Y Screen** | Position Y a l'ecran (pixels). |
| `+0x06` | **X Screen** | Position X a l'ecran (pixels). |

### B. Environnement & Camera

| Variable | Adresse | Description |
| :--- | :--- | :--- |
| **SCX / SCY** | `0xFF42` / `0xFF43` | Scroll X/Y. Permet de compenser le mouvement de la camera. |
| **Tile Map Buffer** | `0xC3A0 - 0xC507` | 360 octets (20 col x 18 lignes) representant l'ecran. |
| **Current Tileset** | `0xD367` | ID du pack de graphismes (ex: `00` overworld, `02` gym). |

---

## 📍 3. Navigation (Overworld)

Donnees pour l'IA de pathfinding (A*).

| Variable | Adresse | Description |
| :--- | :--- | :--- |
| **X / Y Pos** | `0xD362` / `0xD361` | Position actuelle en cases (`0-255`). |
| **Direction** | `0xD35D` | `00` bas, `04` haut, `08` gauche, `0C` droite. |
| **Map ID** | `0xD35E` | ID de la zone (ex: `02` pour Argenta). |
| **Map Connection** | `0xD35F` | `uint8` | Permet de savoir si on change de map. |

Points de passage cles (Map IDs) :

`00`: Bourg Palette | `01`: Jadielle | `02`: Argenta (Pierre) | `51`: Foret de Jade.

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

Lecture securisee des PV (16-bit)

```python
def get_enemy_hp(pyboy):
    # Pokemon stocke les PV sur deux octets (Little Endian)
    low = pyboy.memory[0xCFE6]
    high = pyboy.memory[0xCFE7]
    return low + (high << 8)
```

Conversion pour YOLO

```python
# Exemple pour le joueur depuis C100 (toujours premier slot)
y_pixel = pyboy.memory[0xC104]
x_pixel = pyboy.memory[0xC106]

# Normalisation YOLO (0.0 a 1.0)
y_center = (y_pixel + 8) / 144
x_center = (x_pixel + 8) / 160
```
