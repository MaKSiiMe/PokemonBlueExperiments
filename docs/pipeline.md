# Architecture et Pipeline Technique
## Résolution de Pokémon Rouge par Apprentissage par Renforcement Profond

L'application de l'apprentissage par renforcement profond (Deep Reinforcement Learning, DRL) à des jeux de rôle japonais (JRPG) classiques tels que Pokémon Rouge représente un défi d'ingénierie et de recherche fondamental. Contrairement aux environnements d'arcade traditionnels (Atari 2600) qui se caractérisent par des boucles de rétroaction rapides et des récompenses denses, Pokémon Rouge modélise un **Processus de Décision Markovien Partiellement Observable (POMDP)** d'une complexité redoutable.

Les défis inhérents incluent :
- Des horizons temporels extrêmement longs (dizaines de milliers d'étapes pour atteindre un objectif)
- Des récompenses hautement clairsemées
- Un espace d'états massif (centaines de variables RAM, inventaire, statistiques d'équipe)
- Des transitions d'état non linéaires (rencontres aléatoires, arborescences de dialogues)

Ce document détaille l'architecture, la configuration mathématique et le pipeline technique complet requis pour entraîner un agent autonome capable de maîtriser cet environnement en partant de zéro.

---

## 1. Choix et Configuration de l'Environnement de Simulation

L'interface entre l'algorithme d'apprentissage et le jeu nécessite un environnement de simulation hautement optimisé. L'entraînement d'agents DRL requiert la collecte de dizaines, voire de centaines de millions d'interactions. La vitesse d'émulation et la capacité à extraire des données de manière programmatique sont donc les critères de sélection primordiaux.

### 1.1. Sélection de l'Émulateur — PyBoy

L'émulateur **PyBoy** s'impose comme la solution de référence pour les environnements de type Game Boy en DRL. Contrairement aux émulateurs traditionnels orientés utilisateur final, PyBoy est spécifiquement conçu pour être interfacé avec Python via une API riche. Ses optimisations récentes — compilation à la volée (JIT), meilleure gestion des drapeaux de compilation, utilisation du GIL via Cython — permettent d'atteindre des vitesses dépassant **10 000 étapes par seconde (SPS)** sur des architectures modernes.

L'API permet une injection directe d'inputs et, de manière critique, la lecture et l'écriture directes dans la RAM du jeu via `pyboy.memory[address]`.

### 1.2. Encapsulation via Gymnasium

Pour assurer l'interopérabilité avec les bibliothèques RL (PufferLib, Stable Baselines 3, CleanRL), PyBoy est encapsulé dans une classe héritant de `gymnasium.Env`. Cette encapsulation surcharge les fonctions fondamentales :

| Méthode | Rôle |
|---|---|
| `__init__()` | Initialise l'émulateur, charge la ROM, définit `observation_space` et `action_space` |
| `reset()` | Charge un savestate (`has_pokedex.state`) plaçant directement l'agent dans le monde ouvert avec son premier Pokémon |
| `step(action)` | Applique l'action, avance l'horloge, calcule la récompense, retourne la nouvelle observation |
| `render()` | Extrait le tampon vidéo sous forme de matrice NumPy |

> Le savestate dans `reset()` permet d'éviter que l'agent ne gaspille des millions d'étapes à naviguer dans les menus d'introduction et le tutoriel du Professeur Chen.

### 1.3. Désynchronisation de l'Horloge et Mode Headless

L'émulation doit être découplée du temps réel (60 FPS) :

```python
pyboy.set_emulation_speed(0)  # Exécution aussi rapide que le CPU hôte le permet
```

L'interface utilisateur (SDL/OpenGL) est totalement désactivée (**mode headless**). Le rendu graphique n'est calculé en arrière-plan que lorsqu'une observation visuelle doit être transmise au réseau, minimisant ainsi le CPU-bound bottleneck.

### 1.4. Saut d'Images (Frame Skipping)

La prise de décision à chaque image (1/60e de seconde) est sous-optimale — la majorité des images ne contiennent que des transitions d'animation sans nouvelle information décisionnelle.

Le wrapper exécute l'action choisie, puis avance l'émulateur de **k images consécutives** (généralement `k = 24`) avant de redemander une action à l'agent. La valeur de 24 ticks correspond à la durée d'une animation complète de déplacement d'une tuile dans l'Overworld, synchronisant la cadence de décision de l'IA avec la grille de déplacement fondamentale du jeu.

---

## 2. Définition de l'Espace d'Observation (State Space)

L'architecture optimale utilise un espace hybride (`gymnasium.spaces.Dict`) fusionnant la perception visuelle avec une extraction précise de la sémantique RAM.

### 2.1. Traitement Visuel et Prétraitement des Pixels

**Réduction dimensionnelle et niveaux de gris :**
L'écran natif Game Boy (144 × 160 pixels, palette 4 couleurs) est sous-échantillonné par un facteur 2 → matrice **72 × 80** pixels, convertie en niveaux de gris (1 canal) et normalisée dans $[0, 1]$.

**Empilement Temporel (Frame Stacking) :**
Une seule image statique viole la propriété de Markov (l'agent ne peut déduire la vélocité des entités mobiles). Le wrapper empile la trame courante avec les deux trames précédentes. Le tenseur visuel final a pour dimension **(3, 72, 80)**.

**Masque de Visite (Visited Mask) :**
Un tenseur binaire (matrice rognée **48 × 48** centrée sur l'agent) indique visuellement quelles tuiles ont déjà été explorées durant l'épisode. Sans ce masque, l'agent est frappé d'amnésie spatiale et oscille indéfiniment entre zones adjacentes sans percevoir sa stagnation.

### 2.2. Lecture Directe de la RAM — Memory Map

La Work RAM (WRAM) de Pokémon Rouge contient l'intégralité de l'état logique du jeu. Un sous-ensemble critique est extrait, normalisé et formaté en vecteur scalaire (State Vector).

| Entité Logique | Adresses Hex | Description et Décodage |
|---|---|---|
| Map ID | `C35E` | Entier 8 bits identifiant la carte chargée (ex: `0x00` = Bourg Palette). Converti en One-Hot Encoding ou normalisé. |
| Coordonnées Spatiales | `CC24` (Y) / `CC25` (X) | Positions absolues sur la grille de la carte. Base du calcul des récompenses d'exploration. |
| Points de Vie (Combat) | `D015`-`D016` | PV actuels du Pokémon actif. Format 16 bits Big-Endian. Normalisé en ratio PV/PVmax (÷ `D023`-`D024`). |
| Statut d'Altération | `D018` | Masque de bits : Bit 6 (Paralysie), Bit 5 (Gel), Bit 4 (Brûlure), Bit 3 (Poison), Bits 0-2 (Sommeil). Décomposé en booléens. |
| Statistiques de l'Équipe | `D16B` à `D25C` | Bloc répliqué pour les 6 Pokémon : Espèce (`D16B`), Niveau (`D18C`), PV hors combat (`D16C`-`D16D`). |
| Inventaire | `CF7B` à `CF85` | `CF7B` = nombre d'objets uniques (max 20). Paires (objet, quantité) pour chaque slot. |
| Argent (Pokédollars) | `D347`-`D349` | Format Binary Coded Decimal (BCD). Nécessite un décodage spécifique avant normalisation. |
| Progression Pokédex | `D2F7`-`D31C` | Masques de bits des espèces capturées. Chaque bit à 1 = espèce attrapée. |
| Drapeaux d'Événements | `D7BA` et suivants | Bits globaux (badges, PNJ vaincus, objets uniques). Validation des milestones narratifs. |

Ce vecteur scalaire dense est fusionné avec les représentations visuelles dans l'architecture du réseau, garantissant que l'agent décide non seulement par ce qu'il voit, mais par l'état physiologique de son équipe et l'avancement de sa quête.

---

## 3. Définition de l'Espace d'Actions (Action Space)

### 3.1. Modélisation Mathématique

L'espace est défini comme un espace discret : `gymnasium.spaces.Discrete(7)`.

### 3.2. Correspondance des Touches

| Index | Bouton | Utilisation |
|---|---|---|
| 0 | Flèche Haut | Déplacement, navigation des menus |
| 1 | Flèche Bas | Déplacement, navigation des menus |
| 2 | Flèche Gauche | Déplacement, navigation des menus |
| 3 | Flèche Droite | Déplacement, navigation des menus |
| 4 | Bouton A | Interaction, validation, sélection des attaques |
| 5 | Bouton B | Annulation, fuite, annulation d'évolution |
| 6 | Bouton Start | Ouverture du menu principal |

### 3.3. Omission Stratégique du Bouton Select

Le bouton **Select** est intentionnellement exclu. Son utilité dans Gen 1 est limitée à la réorganisation d'inventaire, gérable via Start. Cette omission réduit l'espace d'exploration de **12,5%**, accélérant mathématiquement la convergence sans pénaliser la capacité de complétion du jeu.

### 3.4. Exécution Asynchrone des Touches

```python
pyboy.button(action)      # Appui
# exécution de 1 tick
pyboy.button_release(action)  # Relâchement explicite (évite la répétition continue)
# exécution des k-1 ticks restants (Frame Skip)
```

---

## 4. Architecture du Modèle (Neural Network)

L'architecture modélise la politique $\pi_\theta(a|s)$ et la fonction de valeur $V_\phi(s)$ avec un design **Actor-Critic** et des extracteurs de caractéristiques distincts pour les données hétérogènes.

### 4.1. Extraction des Caractéristiques Visuelles (CNN)

Le tenseur visuel (dimension **3 × 72 × 80**) et le masque de visite (dimension **1 × 48 × 48**) sont traités par des CNN de type **Nature CNN** (architecture DeepMind, robustesse prouvée sur Atari).

| Couche | Filtres | Taille du Noyau | Stride | Activation |
|---|---|---|---|---|
| Conv1 | 32 | 8 × 8 | 4 | ReLU |
| Conv2 | 64 | 4 × 4 | 2 | ReLU |
| Conv3 | 64 | 3 × 3 | 1 | ReLU |
| Flatten | — | — | — | → vecteur 1D |

Un CNN parallèle, potentiellement plus léger, traite le masque binaire de visite.

### 4.2. Traitement du Vecteur d'État Logique (MLP)

Le vecteur RAM (statistiques, niveaux, booléens de statut, identifiants d'objets) n'a pas de structure spatiale et est traité par un **MLP** :

- 2 couches denses de **256 neurones** chacune
- Activation ReLU

**Fusion des Modalités :** Les encodages du CNN (écran), du CNN (masque de visite) et du MLP (RAM) sont concaténés en un vecteur d'intégration globale $E_t$.

### 4.3. Gestion de la Mémoire à Long Terme (Couche Récurrente)

Dans Pokémon, les informations disparaissent fréquemment de l'écran (ex: un combat efface la vision de la carte Overworld). Une mémoire de travail robuste est indispensable.

**Choix architectural — GRU vs LSTM vs Transformer :**

Les architectures Transformer offrent théoriquement une capacité supérieure sur de longues dépendances, mais leur complexité est **quadratique** $O(N^2)$ par rapport à la longueur de séquence $N$. Sur des épisodes de 100 000 étapes, l'empreinte mémoire devient prohibitive. Les **GRU** et **LSTM** (états cachés de taille fixe) restent l'état de l'art optimal pour ce cas d'usage.

**Implémentation :** Couche **GRU** (plus légère qu'un LSTM, performances empiriques similaires) avec une taille cachée de **512 neurones** :

$$h_t = \text{GRU}(E_t, h_{t-1})$$

### 4.4. Têtes de Sortie (Actor-Critic Heads)

Le vecteur récurrent $h_t$ alimente simultanément deux têtes linéaires :

**Acteur (Policy Head)** $\pi_\theta(a_t | s_t, h_t)$
: Projection vers 7 neurones + Softmax → distribution de probabilités sur les actions.

**Critique (Value Head)** $V_\phi(s_t, h_t)$
: Projection vers 1 neurone scalaire → estimation du Return escompté depuis cet état.

---

## 5. Stratégie d'Exploration et Algorithme RL

### 5.1. Proximal Policy Optimization (PPO)

L'algorithme choisi est **PPO**, standard de facto pour les environnements complexes. Contrairement à DQN (off-policy, instable dans les environnements stochastiques comme les combats soumis au RNG), PPO est une méthode **on-policy de gradient de politique** qui optimise directement les probabilités des actions.

**Stabilité par Écrêtage (Clipping) :**

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$

Où :
- $r_t(\theta) = \dfrac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$ est le ratio de politique
- $\epsilon \approx 0.2$ est l'hyperparamètre d'écrêtage

**Avantage Généralisé (GAE) :**
Le terme $\hat{A}_t$ indique si l'action sélectionnée a donné de meilleurs résultats que la prédiction du Critique. Si $\hat{A}_t > 0$, la probabilité de l'action est augmentée.

### 5.2. Résolution du Problème d'Exploration — Go-Explore

Les méthodes de curiosité pure (RND, Intrinsic Motivation) souffrent de deux problèmes majeurs :
- **Noisy TV Problem** : l'agent fixe des éléments stochastiques (animations de l'eau, PNJ aléatoires) pour maximiser la nouveauté perçue
- **Détachement (Detachment)** : l'agent oublie les zones prometteuses explorées précédemment

Le pipeline intègre une architecture inspirée de **Go-Explore** (popularisée par la résolution de Montezuma's Revenge).

**L'Archive Sémantique (Cell Archive) :**
Une base de données stockant des "Cellules" définies par l'empreinte combinée : `Map ID (C35E)` + coordonnées `X/Y (CC24/CC25)`. Chaque cellule conserve le savestate PyBoy correspondant.

**Phase — Retourner et Explorer :**
Au lieu de toujours recommencer à Bourg Palette, l'agent sélectionne probabilistiquement une cellule dans l'archive (en favorisant les frontières de l'exploration connue), charge instantanément le savestate PyBoy associé, et reprend l'exploration depuis ce point d'ancrage avancé.

Cette stratégie de *save-scumming algorithmique* force la politique à s'étendre continuellement vers de nouvelles zones sans réapprendre les dizaines de milliers d'actions préalables.

---

## 6. Ingénierie de la Fonction de Récompense (Reward Shaping)

### 6.1. Formule Globale

La récompense totale fournie à chaque transition d'état est :

$$R_{total} = R_{event} + R_{nav} + R_{heal} + R_{level}$$

**Récompense d'Événement** $R_{event}$
: Signal extrinsèque le plus fort. Chaque fois qu'un bit critique dans la table `D7BA` passe de 0 à 1 (ex: obtention du Pokédex, victoire contre un champion d'arène), une récompense massive est émise ($+2.0$ à $+4.0$).

**Récompense de Navigation** $R_{nav}$
: Chaque fois que les coordonnées `(X, Y, Map ID)` n'ont jamais été visitées dans le mini-épisode courant, une récompense de $+0.005$ est accordée. Ce signal dense, corrélé à Go-Explore, force l'agent à couvrir géométriquement les cartes.

**Récompense de Survie** $R_{heal}$
: Une augmentation abrupte des PV globaux de l'équipe (Potion utilisée, infirmière du Centre Pokémon) génère une récompense calibrée positive.

**Récompense de Niveau** $R_{level}$
: Formule affine par morceaux inspirée des travaux de Peter Whidden :

$$R_{level} = \begin{cases} \displaystyle\sum_{i=1}^{6} level_i & \text{si } \displaystyle\sum_{i=1}^6 level_i < 15 \\[10pt] 30 + \dfrac{\displaystyle\sum_{i=1}^{6} level_i - 15}{4} & \text{si } \displaystyle\sum_{i=1}^6 level_i \ge 15 \end{cases}$$

Cette logique est délibérément asymptotique : elle récompense équitablement chaque niveau jusqu'à la somme 15 (incitant à entraîner le Pokémon de départ), puis le rendement marginal décroît fortement (÷4). Cette pénalité empêche l'agent de stagner en Route 1 à grinder des Roucool de bas niveau, le forçant à chercher les récompenses d'événement plus lucratives.

### 6.2. Analyse du Reward Hacking et Échecs Documentés

Les réseaux de neurones exploitent la moindre faille dans la fonction de récompense — un phénomène appelé **Reward Hacking**. Les itérations passées ont mis en lumière plusieurs de ces failles.

---

**Le Traumatisme du PC**

**Problème :** Si l'agent dépose un Pokémon de haut niveau dans l'ordinateur du Centre Pokémon, $\sum level_i$ s'effondre. PPO perçoit cette chute comme une punition colossale. L'IA développe une "phobie" des Centres Pokémon et refuse d'y entrer, laissant ses Pokémon mourir.

**Solution :** Lire simultanément la RAM des boîtes PC (SRAM banks 2-3, `A000`-`BFFF`) pour intégrer les Pokémon stockés dans la somme des niveaux, ou asymétriser la récompense (autoriser la hausse, censurer la chute liée à un dépôt volontaire).

---

**L'Exploitation du Casino de Céladopole**

**Problème :** Des micro-récompenses liées à l'appui du bouton A face à des objets interactifs ont conduit l'agent à rester bloqué sur les machines à sous, marteler le bouton et engranger des micro-récompenses à l'infini.

**Solution :** Plafonner les micro-récompenses par épisode, ou les indexer de manière décroissante selon le nombre d'interactions répétées sur la même tuile.

---

**Le Faux Positif d'Exploration (KNN Animation)**

**Problème :** Les premières versions évaluaient l'exploration via la nouveauté visuelle des pixels (KNN ou réseau de distillation). L'agent apprenait à fixer des objets animés (eau, PNJ stochastiques) pour générer un flux infini de "nouveauté".

**Solution :** La récompense $R_{nav}$ actuelle repose strictement sur les coordonnées hexadécimales extraites de la RAM (`CC24`/`CC25`), ignorant totalement le bruit visuel dans la fonction de coût.

---

## 7. Infrastructure, Parallélisation et Déploiement

### 7.1. Parallélisation avec PufferLib

L'émulation PyBoy est fondamentalement **CPU-Bound**. L'évaluation séquentielle d'un seul environnement ne génère qu'une fraction des données nécessaires à saturer un GPU moderne.

**Instances multiples :** Entre **32 et 96 instances parallèles** (`num_envs=96`) réparties sur l'ensemble des cœurs disponibles.

**Goulot d'étranglement de Gymnasium :** L'API native (`SyncVectorEnv`, `AsyncVectorEnv`) introduit une surcharge IPC importante via `multiprocessing` Python.

**Intégration PufferLib :** PufferLib substitue la vectorisation native par un backend ultra-optimisé en C, utilisant de la **mémoire partagée** (Shared Memory) pour fusionner directement les observations des 96 instances PyBoy en tenseurs préalloués. Ces tenseurs denses sont expédiés en bloc (Pinned Memory) vers le GPU. Combiné à `torch.compile`, cette architecture permet d'atteindre des pics de **10 000 à 30 000 SPS** selon le matériel.

### 7.2. Besoins Matériels et Tampons de Trajectoires

**Hardware :**
- **CPU :** Processeur à haut parallélisme (ex: AMD Ryzen 9 / Threadripper, 16-32 cœurs) pour la simulation asynchrone des 96 émulateurs.
- **GPU :** Le modèle Actor-Critic PPO + GRU est extrêmement léger (< 10 millions de paramètres, ~60 500× plus petit que DeepSeekV3). Une carte grand public performante (ex: NVIDIA RTX 4090) est largement suffisante.

**Rollout Buffers :**
PPO est un algorithme on-policy — il n'utilise pas de Replay Buffer massif. Il utilise un tampon de trajectoires alloué typiquement sur **2048 ou 4096 étapes par environnement** :

$$\text{Batch Size} = 96 \text{ envs} \times 4096 \text{ étapes} = 393\ 216 \text{ transitions}$$

Une fois le GPU ayant optimisé le réseau sur ce lot via rétropropagation temporelle (BPTT pour les couches GRU), le buffer est vidé et l'échantillonnage reprend.

### 7.3. Monitoring et Métriques Critiques

Un entraînement de plusieurs dizaines d'heures nécessite un suivi télémétrique rigoureux via **TensorBoard** ou **Weights & Biases (W&B)**.

**Métriques de l'Algorithme RL :**

| Métrique | Interprétation |
|---|---|
| Entropie de la Politique | Un effondrement vers 0 indique que l'agent a cessé d'explorer et s'est enfermé dans un optimum local (ex: tourner en rond) |
| Perte de Valeur (Value Loss) | Évalue la précision du Critic à prédire les récompenses futures. Une Value Loss stable confirme l'ajustement au Reward Shaping |
| Débit (SPS) | Détecte l'étranglement silencieux du CPU |

**Métriques du Jeu :**

| Métrique | Interprétation |
|---|---|
| Cartes Uniques (Unique Map IDs) | Mesure de progression macroscopique. Les pics correspondent à la découverte de nouvelles villes |
| Niveau Max Atteint | Vérifie que l'agent apprend les boucles d'engagement de combat et soigne ses Pokémon |
| Taux de Complétion des Jalons | Courbes traçant la fréquence des franchissements critiques (Bourg Palette → Forêt de Jade → Badge Pierre → Mont Sélénite) sur les 96 environnements parallèles. Valident la résilience de la politique face à l'étendue globale du JRPG |
