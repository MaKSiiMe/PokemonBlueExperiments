# 🧠 RAM Map (Pokémon Blue US/EU)

Ce document recense les adresses mémoires critiques identifiées pour le fonctionnement de l'agent.

> **Note Importante :** Les adresses correspondent à la version **US/EU (Anglaise)** de la ROM. Le projet a migré depuis la version FR pour garantir la stabilité des adresses mémoires standards.

---

## 📍 Navigation (Positionnement)

Ces variables permettent de situer le joueur dans le monde global.

| Variable | Adresse (Hex) | Type | Description |
| :--- | :--- | :--- | :--- |
| **X Position** | `0xD362` | `uint8` | Coordonnée X du joueur sur la map actuelle (en cases). |
| **Y Position** | `0xD361` | `uint8` | Coordonnée Y du joueur sur la map actuelle (en cases). |
| **Map ID** | `0xD35E` | `uint8` | Identifiant unique de la zone actuelle. |

### Map IDs Connus

| ID | Zone | Notes |
| :---: | :--- | :--- |
| 0 | Bourg Palette | Zone de départ |
| 1 | Jadielle (Viridian City) | Première ville |
| 2 | Argenta (Pewter City) | Arène de Pierre (Brock) |
| 12 | Route 1 | Entre Palette et Jadielle |
| 13 | Route 2 | Vers la Forêt de Jade |
| 33 | Route 22 | Vers la Ligue Pokémon |
| 51 | Forêt de Jade | Labyrinthe |

---

## 🎥 Vision & Graphismes (Hardware Registers)

Ces registres matériels sont utilisés par le module de Computer Vision pour la **calibration dynamique** de la caméra et la détection des sprites.

| Variable | Adresse (Hex) | Type | Description |
| :--- | :--- | :--- | :--- |
| **SCY (Scroll Y)** | `0xFF42` | `uint8` | Position verticale de la caméra (Viewport) en pixels. |
| **SCX (Scroll X)** | `0xFF43` | `uint8` | Position horizontale de la caméra (Viewport) en pixels. |
| **OAM (Sprites)** | `0xFE00 - 0xFE9F` | Range | Mémoire vidéo contenant les 40 sprites. |

### Structure OAM (Object Attribute Memory)

Chaque sprite occupe **4 octets** :

| Offset | Donnée | Description |
| :---: | :--- | :--- |
| +0 | Y Position | Position Y écran + 16 |
| +1 | X Position | Position X écran + 8 |
| +2 | Tile ID | Index de la tuile graphique |
| +3 | Attributes | Flags (palette, flip, priorité) |

**Exemple de lecture :**
```python
OAM_BASE = 0xFE00
for i in range(40):  # 40 sprites max
    addr = OAM_BASE + (i * 4)
    y = pyboy.memory[addr] - 16      # Correction offset Y
    x = pyboy.memory[addr + 1] - 8   # Correction offset X
    tile_id = pyboy.memory[addr + 2]
    attributes = pyboy.memory[addr + 3]
```

---

## ⚔️ Combat (Battle State)

Variables utilisées pour détecter l'état de combat et calculer les récompenses.

| Variable | Adresse (Hex) | Type | Description |
| :--- | :--- | :--- | :--- |
| **Battle Status** | `0xD057` | `uint8` | État du jeu (voir tableau ci-dessous). |
| **Enemy HP** | `0xCFE6` | `uint16` | PV actuels de l'adversaire. |
| **My HP** | `0xD16C` | `uint16` | PV actuels du Pokémon actif. |

### Valeurs de Battle Status

| Valeur | État | Description |
| :---: | :--- | :--- |
| 0 | Overworld | Exploration normale |
| 1 | Wild Battle | Combat contre un Pokémon sauvage |
| 2 | Trainer Battle | Combat contre un dresseur |

---

## 🎒 Inventaire & Progression

| Variable | Adresse (Hex) | Type | Description |
| :--- | :--- | :--- | :--- |
| **Badges** | `0xD356` | `uint8` | Badges obtenus (Bitmask) |
| **Money** | `0xD347` | `uint24` | Argent (BCD, 3 octets) |
| **Pokédex Owned** | `0xD2F7` | `uint8` | Nombre de Pokémon possédés |

---

## 🧪 Notes Techniques

### Accès Mémoire

```python
# Lecture d'un octet
value = pyboy.memory[0xD362]

# Lecture d'un mot (16 bits, Little Endian)
low = pyboy.memory[0xCFE6]
high = pyboy.memory[0xCFE7]
value_16bit = low + (high << 8)
```

### Little Endian

La Game Boy stocke les données multi-octets **à l'envers** (octet de poids faible en premier) :

```
Adresse:  0xCFE6  0xCFE7
Contenu:  [0x1A]  [0x00]
Valeur:   0x001A = 26 (décimal)
```

### Conversion Pixels ↔ Cases

Le jeu utilise des **cases de 16x16 pixels** :

```python
# Position en pixels
pixel_x = case_x * 16
pixel_y = case_y * 16

# Position en cases
case_x = pixel_x // 16
case_y = pixel_y // 16
```

---

## 📚 Ressources

- [pret/pokered](https://github.com/pret/pokered) - Désassemblage officiel de Pokémon Rouge/Bleu
- [Data Crystal - Pokemon Red/Blue](https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue) - Wiki ROM Hacking
