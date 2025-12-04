# 👁️ Vision Pipeline: From RAM to YOLO

Ce document détaille l'architecture du système de **Computer Vision** utilisé pour permettre à l'agent de "voir" le jeu.

> **Philosophie :** Contrairement aux approches classiques qui lisent directement la RAM pour prendre des décisions, nous utilisons la RAM uniquement pour **générer la vérité terrain (Ground Truth)**, afin d'entraîner un réseau de neurones capable de généraliser.

---

## 1. Le Défi Technique

La Game Boy n'a pas de concept d'"Objets" au sens moderne :

| Élément | Stockage | Problème |
| :--- | :--- | :--- |
| **Personnages** | Sprites OAM (4 tuiles 8x8) | Un perso = 4 sprites séparés |
| **Portes/Panneaux** | Background Tiles | Indistinguables visuellement du décor |
| **Caméra** | Registres SCX/SCY | Scrolling continu, calcul de position nécessaire |

---

## 2. Data Engineering : La Génération du Dataset

Script principal : `src/vision/generate_dataset.py`

### A. Algorithme de Clustering (Détection des Sprites)

**Problème :** Un personnage dans l'OAM = 4 sprites 8x8 séparés.

**Solution :** Algorithme de regroupement spatial.

```
┌────────────────────────────────────────────────┐
│  OAM Memory (0xFE00 - 0xFE9F)                  │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐                       │
│  │8x8│ │8x8│ │8x8│ │8x8│  ← 4 sprites séparés  │
│  └───┘ └───┘ └───┘ └───┘                       │
│       ↓ Clustering (dist < 12px)               │
│  ┌───────────────┐                             │
│  │    16x16      │  ← 1 entité fusionnée       │
│  │   (Player)    │                             │
│  └───────────────┘                             │
└────────────────────────────────────────────────┘
```

**Paramètres :**
- Seuil de distance : `12 pixels`
- Taille finale : `16x16 pixels`
- Filtre Tile ID : `< 0xF0` (ignore les tuiles utilitaires)

### B. Identification Joueur vs NPC

**Heuristique :** Le joueur est toujours au centre de l'écran (80, 72).

```python
center_x, center_y = 80, 72  # Centre écran Game Boy (160x144)
player = min(entities, key=lambda e: distance(e.center, (80, 72)))
```

**Garde-fou :** Si la distance minimale > 70px, la frame est ignorée (cas rare, ex: animation de saut).

### C. Placement des Objets Statiques

Les objets fixes (portes, panneaux) sont placés en fonction de la position du joueur :

```python
# Position monde du joueur (depuis la RAM, en cases)
player_world_x = pyboy.memory[0xD362]
player_world_y = pyboy.memory[0xD361]

# Position écran d'un objet = différence avec le joueur (en pixels)
# Le joueur étant toujours au centre (80, 72)
door_screen_x = 80 + (door_world_x - player_world_x) * 16
door_screen_y = 72 + (door_world_y - player_world_y) * 16
```

### D. Atlas des Objets Statiques

Les objets fixes (portes, panneaux) sont répertoriés manuellement par Map ID :

```python
KNOWN_DOORS = {
    0:  [(5, 5), (13, 5), (12, 11)],   # Bourg Palette
    1:  [(23, 25), (29, 19), ...],     # Jadielle
    2:  [(13, 25), (23, 17), ...],     # Argenta
    33: [(6, 5)]                        # Route 22
}

KNOWN_SIGNS = {
    0: [(3, 5), (11, 5), (7, 9), ...], # Palette
    1: [(13, 19), (19, 27)],           # Jadielle
    # ...
}
```

### E. Diversité du Dataset

Pour éviter l'overfitting, le dataset est généré à partir de **~50 Save States** :

| Type de zone | Exemples |
| :--- | :--- |
| Intérieurs | Maisons, Labo Chen, Centre Pokémon |
| Extérieurs | Bourg Palette, Jadielle, Routes |
| Zones spéciales | Forêt de Jade (sombre), Hautes herbes |

---

## 3. Format de Sortie (YOLO)

### Structure des fichiers

```
data/dataset/raw/
├── images/
│   ├── train_00000.jpg
│   ├── train_00001.jpg
│   └── ...
└── labels/
    ├── train_00000.txt
    ├── train_00001.txt
    └── ...
```

### Format des labels (YOLO)

```
<class_id> <x_center> <y_center> <width> <height>
```

Toutes les valeurs sont **normalisées** entre 0 et 1.

**Exemple :**
```
0 0.500000 0.500000 0.100000 0.111111  # Player au centre
1 0.750000 0.300000 0.100000 0.111111  # NPC en haut à droite
2 0.200000 0.600000 0.100000 0.111111  # Door en bas à gauche
```

---

## 4. Le Modèle YOLOv8

### Configuration

| Paramètre | Valeur |
| :--- | :--- |
| **Architecture** | YOLOv8-Nano |
| **Input** | 320x320 RGB (upscaled depuis 160x144) |
| **Classes** | 4 (Player, NPC, Door, Sign) |
| **Epochs** | 50 |
| **Batch Size** | 16 |

### Fichier de configuration (`data.yaml`)

```yaml
path: datasets
train: train/images
val: val/images

names:
  0: Player
  1: NPC
  2: Door
  3: Sign
```

### Performances (Validation)

| Métrique | Score |
| :--- | :---: |
| **mAP50** | > 99.2% |
| **mAP50-95** | > 97% |
| **Precision** | > 99% |
| **Recall** | > 99% |
| **Inférence** | ~3ms/frame (RTX 3060) |

---

## 5. Pipeline d'Inférence (Temps Réel)

### Flux de données

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  PyBoy   │───►│  Convert │───►│  YOLO    │───►│  Agent   │
│  Screen  │    │  RGB→BGR │    │  Predict │    │  Input   │
│  (RGBA)  │    │  NumPy   │    │  Boxes   │    │  Vector  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### Code d'inférence

```python
# Capture
screen = pyboy.screen.image
if screen.mode == 'RGBA':
    screen = screen.convert('RGB')

# Conversion pour OpenCV
frame = np.array(screen)
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Inférence
results = model(frame, imgsz=320, conf=0.5, verbose=False)

# Extraction des détections
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
```

### Couleurs de visualisation (BGR)

| Classe | Couleur | Code BGR |
| :--- | :--- | :--- |
| Player | Bleu | `(255, 0, 0)` |
| NPC | Rouge | `(0, 0, 255)` |
| Door | Jaune | `(0, 255, 255)` |
| Sign | Vert | `(0, 255, 0)` |

---

## 6. Scripts Utilitaires

| Script | Description |
| :--- | :--- |
| `generate_dataset.py` | Génère 5000 images annotées |
| `split_data.py` | Split train/val (80/20) |
| `visualize_dataset.py` | Debug : dessine les boxes sur les images |
| `test_inference.py` | Test temps réel avec PyBoy |

### Commandes utiles

```bash
# Nettoyer le dataset sans supprimer les dossiers
find data/dataset -type f -delete

# Générer un nouveau dataset
python3 src/vision/generate_dataset.py

# Vérifier les annotations (debug)
python3 src/vision/visualize_dataset.py
```