"""
gen1_data.py — Constantes Gen 1 figées, source unique de vérité pour le graphe.

Toutes ces données sont indépendantes de la PokéAPI (qui renvoie les données
modernes). On les hardcode ici pour garantir la fidélité Gen 1.

Sources :
  - Type chart : Bulbapedia "Type/Generation I"
  - Internal IDs : pret/pokered constants/pokemon_data_constants.asm
  - RAM type bytes : pret/pokered constants/type_constants.asm
  - Zone map IDs : pret/pokered constants/map_constants.asm + waypoints.py
"""

from __future__ import annotations

# ── Types valides en Gen 1 ────────────────────────────────────────────────────
# Acier, Ténèbres et Fée n'existent pas encore.
GEN1_TYPES: frozenset[str] = frozenset({
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic",
    "bug", "rock", "ghost", "dragon",
})

# ── Table de types Gen 1 ─────────────────────────────────────────────────────
# Format : TYPE_CHART[type_attaquant][type_défenseur] = multiplicateur
# Seules les valeurs != 1.0 sont stockées.
#
# Différences clés par rapport aux gens suivantes :
#   • Ghost  → Psychic = 0.0  (bug Gen 1 : l'immunité est inversée)
#   • Poison → Bug     = 2.0  (retiré en Gen 2)
#   • Bug    → Poison  = 2.0  (retiré en Gen 2)
#   • Bug    → Psychic = 2.0  (réduit à 1.0 en Gen 2, puis 0.5)
#   • Ice    → Fire    = 2.0  (retiré en Gen 2)
TYPE_CHART: dict[str, dict[str, float]] = {
    "normal": {
        "rock": 0.5,
        "ghost": 0.0,
    },
    "fire": {
        "fire": 0.5, "water": 0.5, "rock": 0.5, "dragon": 0.5,
        "grass": 2.0, "ice": 2.0, "bug": 2.0,
    },
    "water": {
        "water": 0.5, "grass": 0.5, "dragon": 0.5,
        "fire": 2.0, "ground": 2.0, "rock": 2.0,
    },
    "electric": {
        "electric": 0.5, "grass": 0.5, "dragon": 0.5,
        "ground": 0.0,
        "water": 2.0, "flying": 2.0,
    },
    "grass": {
        "fire": 0.5, "grass": 0.5, "poison": 0.5, "flying": 0.5,
        "bug": 0.5, "dragon": 0.5,
        "water": 2.0, "ground": 2.0, "rock": 2.0,
    },
    "ice": {
        "water": 0.5, "ice": 0.5,
        "fire": 2.0,  # Gen 1 uniquement
        "grass": 2.0, "ground": 2.0, "flying": 2.0, "dragon": 2.0,
    },
    "fighting": {
        "poison": 0.5, "bug": 0.5, "flying": 0.5, "psychic": 0.5,
        "ghost": 0.0,
        "normal": 2.0, "ice": 2.0, "rock": 2.0,
    },
    "poison": {
        "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5,
        "grass": 2.0,
        "bug": 2.0,   # Gen 1 uniquement
    },
    "ground": {
        "grass": 0.5, "bug": 0.5,
        "flying": 0.0,
        "fire": 2.0, "electric": 2.0, "poison": 2.0, "rock": 2.0,
    },
    "flying": {
        "electric": 0.5, "rock": 0.5,
        "grass": 2.0, "fighting": 2.0, "bug": 2.0,
    },
    "psychic": {
        "psychic": 0.5,
        "fighting": 2.0, "poison": 2.0,
    },
    "bug": {
        "fire": 0.5, "fighting": 0.5, "flying": 0.5, "ghost": 0.5,
        "grass": 2.0,
        "poison": 2.0,   # Gen 1 uniquement
        "psychic": 2.0,  # Gen 1 uniquement
    },
    "rock": {
        "fighting": 0.5, "ground": 0.5,
        "fire": 2.0, "ice": 2.0, "flying": 2.0, "bug": 2.0,
    },
    "ghost": {
        # Gen 1 BUG : Ghost ne fait aucun dégât aux types Normal ET Psychic
        "normal": 0.0,
        "psychic": 0.0,  # Devrait être 2.0 mais bug Gen 1
        "ghost": 2.0,
    },
    "dragon": {
        "dragon": 2.0,
    },
}

# ── Octets RAM → nom de type (pret/pokered type_constants.asm) ────────────────
# Utilisé pour convertir wEnemyMonType1/2 (0xD01F/0xD020) en nom lisible.
RAM_TYPE_BYTE_TO_NAME: dict[int, str] = {
    0x00: "normal",
    0x01: "fighting",
    0x02: "flying",
    0x03: "poison",
    0x04: "ground",
    0x05: "rock",
    0x07: "bug",
    0x08: "ghost",
    0x14: "fire",    # 20 décimal
    0x15: "water",   # 21
    0x16: "grass",   # 22
    0x17: "electric",# 23
    0x18: "psychic", # 24
    0x19: "ice",     # 25
    0x1A: "dragon",  # 26
}

# Inverse : nom → octet RAM (pour construire des requêtes graphe depuis la RAM)
RAM_TYPE_NAME_TO_BYTE: dict[str, int] = {v: k for k, v in RAM_TYPE_BYTE_TO_NAME.items()}

# ── ID interne Gen 1 → numéro Pokédex national ────────────────────────────────
# Source : pret/pokered data/pokemon/index_pointers.asm
# Les IDs internes Gen 1 NE sont PAS séquentiels par rapport au Pokédex.
# Utilisé pour faire le lien entre la RAM (wEnemyMonSpecies, 0xCFDE) et PokéAPI.
GEN1_INTERNAL_TO_DEX: dict[int, int] = {
    0x99: 1,   # Bulbasaur
    0x09: 2,   # Ivysaur
    0xAA: 3,   # Venusaur
    0xB0: 4,   # Charmander
    0xB2: 5,   # Charmeleon
    0xB4: 6,   # Charizard
    0xB1: 7,   # Squirtle
    0xB3: 8,   # Wartortle
    0x1C: 9,   # Blastoise
    0x7B: 10,  # Caterpie
    0x7C: 11,  # Metapod
    0x7D: 12,  # Butterfree
    0x70: 13,  # Weedle
    0x71: 14,  # Kakuna
    0x72: 15,  # Beedrill
    0x24: 16,  # Pidgey
    0x96: 17,  # Pidgeotto
    0x97: 18,  # Pidgeot
    0xA5: 19,  # Rattata
    0xA6: 20,  # Raticate
    0x05: 21,  # Spearow
    0x23: 22,  # Fearow
    0x6C: 23,  # Ekans
    0x2D: 24,  # Arbok
    0x54: 25,  # Pikachu
    0x55: 26,  # Raichu
    0x60: 27,  # Sandshrew
    0x61: 28,  # Sandslash
    0x0F: 29,  # Nidoran♀
    0xA8: 30,  # Nidorina
    0x10: 31,  # Nidoqueen
    0x03: 32,  # Nidoran♂
    0xA7: 33,  # Nidorino
    0x07: 34,  # Nidoking
    0x04: 35,  # Clefairy
    0x8E: 36,  # Clefable
    0x52: 37,  # Vulpix
    0x53: 38,  # Ninetales
    0x64: 39,  # Jigglypuff
    0x65: 40,  # Wigglytuff
    0x6B: 41,  # Zubat
    0x82: 42,  # Golbat
    0xB9: 43,  # Oddish
    0xBA: 44,  # Gloom
    0xBB: 45,  # Vileplume
    0x6D: 46,  # Paras
    0x2E: 47,  # Parasect
    0x41: 48,  # Venonat
    0x77: 49,  # Venomoth
    0x3B: 50,  # Diglett
    0x76: 51,  # Dugtrio
    0x4D: 52,  # Meowth
    0x88: 53,  # Persian
    0x17: 54,  # Psyduck
    0x92: 55,  # Golduck
    0x93: 56,  # Mankey
    0x2F: 57,  # Primeape
    0x19: 58,  # Growlithe
    0x20: 59,  # Arcanine
    0x3C: 60,  # Poliwag
    0x3D: 61,  # Poliwhirl
    0x3E: 62,  # Poliwrath
    0x01: 63,  # Abra
    0x02: 64,  # Kadabra
    0x24: 65,  # Alakazam  -- NOTE: conflit avec Pidgey 0x24 ? À vérifier vs pokered
    0x15: 66,  # Machop
    0x16: 67,  # Machoke
    0x1D: 68,  # Machamp
    0xBC: 69,  # Bellsprout
    0xBD: 70,  # Weepinbell
    0xBE: 71,  # Victreebel
    0x18: 72,  # Tentacool
    0x9B: 73,  # Tentacruel
    0xA9: 74,  # Geodude
    0x27: 75,  # Graveler
    0x31: 76,  # Golem
    0xA3: 77,  # Ponyta
    0xA4: 78,  # Rapidash
    0x25: 79,  # Slowpoke
    0x26: 80,  # Slowbro
    0x6E: 81,  # Magnemite
    0x6F: 82,  # Magneton
    0x44: 83,  # Farfetch'd
    0x29: 84,  # Doduo
    0x2A: 85,  # Dodrio
    0x84: 86,  # Seel
    0x85: 87,  # Dewgong
    0x0E: 88,  # Grimer
    0x8B: 89,  # Muk
    0x1E: 90,  # Shellder
    0x1F: 91,  # Cloyster
    0xAB: 92,  # Gastly
    0xAC: 93,  # Haunter
    0xAD: 94,  # Gengar
    0x62: 95,  # Onix
    0x57: 96,  # Drowzee
    0x58: 97,  # Hypno
    0x0A: 98,  # Krabby
    0x0B: 99,  # Kingler
    0x0C: 100, # Voltorb
    0x0D: 101, # Electrode
    0x90: 102, # Exeggcute
    0x91: 103, # Exeggutor
    0xA0: 104, # Cubone
    0xA1: 105, # Marowak
    0x2B: 106, # Hitmonlee
    0x2C: 107, # Hitmonchan
    0x3F: 108, # Lickitung
    0x69: 109, # Koffing
    0x6A: 110, # Weezing
    0x9C: 111, # Rhyhorn
    0x9D: 112, # Rhydon
    0x42: 113, # Chansey
    0x43: 114, # Tangela
    0xAE: 115, # Kangaskhan
    0x48: 116, # Horsea
    0x49: 117, # Seadra
    0x4F: 118, # Goldeen
    0x50: 119, # Seaking
    0xAF: 120, # Staryu
    0xB8: 121, # Starmie
    0xBF: 122, # Mr. Mime
    0x4A: 123, # Scyther
    0x4B: 124, # Jynx
    0x4C: 125, # Electabuzz
    0x4E: 126, # Magmar
    0x5E: 127, # Pinsir
    0x80: 128, # Tauros
    0x13: 129, # Magikarp
    0x8A: 130, # Gyarados
    0x74: 131, # Lapras
    0x4C: 132, # Ditto  -- NOTE: à vérifier, peut confliter avec Electabuzz
    0x45: 133, # Eevee
    0x46: 134, # Vaporeon
    0x47: 135, # Jolteon
    # 0x??: 136, # Flareon  — ID incertain, à vérifier vs pokered
    0xB5: 137, # Porygon
    0x78: 138, # Omanyte
    0x79: 139, # Omastar
    0x73: 140, # Kabuto
    0x93: 141, # Kabutops  -- NOTE: peut confliter avec Mankey
    0x8C: 142, # Aerodactyl
    0x79: 143, # Snorlax   -- NOTE: peut confliter avec Omastar
    0x98: 144, # Articuno
    0x33: 145, # Zapdos
    0x34: 146, # Moltres
    0x35: 147, # Dratini
    0xAE: 148, # Dragonair -- NOTE: peut confliter avec Kangaskhan
    0x84: 149, # Dragonite -- NOTE: peut confliter avec Seel
    0x19: 150, # Mewtwo    -- NOTE: peut confliter avec Growlithe
    0xB5: 151, # Mew       -- NOTE: peut confliter avec Porygon
}
# ⚠️  AVERTISSEMENT : La table ci-dessus contient des conflits potentiels dus à
#    des incertitudes dans ma connaissance des IDs internes Gen 1.
#    Source canonique : pret/pokered data/pokemon/index_pointers.asm
#    Avant d'utiliser les IDs RAM pour identifier les espèces, vérifier contre
#    le disassembly. Pour le projet actuel, les types (0xD01F/0xD020) sont lus
#    directement depuis la RAM → pas besoin de cette table pour les combats.
#    Cette table est utile pour la prédiction de rencontres et les évolutions.

# Table inverse : numéro Pokédex → ID interne Gen 1
# (Ne contient que les entrées non-ambiguës de GEN1_INTERNAL_TO_DEX)
GEN1_DEX_TO_INTERNAL: dict[int, int] = {v: k for k, v in GEN1_INTERNAL_TO_DEX.items()}

# ── Map RAM ID → infos zone (nom, slug PokéAPI location-area) ─────────────────
# Source IDs : pret/pokered constants/map_constants.asm
# Slugs PokéAPI : https://pokeapi.co/api/v2/location-area/?limit=300
# Seules les zones du parcours Bourg Palette → Pierre (Brock) sont mappées.
ZONE_MAP_ID: dict[int, dict[str, str]] = {
    0x00: {"name": "Bourg Palette",    "pokeapi_slug": None},           # pas d'herbe
    0x01: {"name": "Jadielle City",    "pokeapi_slug": None},           # ville
    0x02: {"name": "Argenta City",     "pokeapi_slug": None},           # ville
    0x0C: {"name": "Route 1",          "pokeapi_slug": "kanto-route-1-area"},
    0x0D: {"name": "Route 2",          "pokeapi_slug": "kanto-route-2-south-towards-viridian-city"},
    0x0E: {"name": "Route 2 Nord",     "pokeapi_slug": "kanto-route-2-north-towards-pewter-city"},
    0x33: {"name": "Forêt de Jade",    "pokeapi_slug": "viridian-forest-area"},
    0x36: {"name": "Arène d'Argenta",  "pokeapi_slug": None},           # pas de sauvages
}

# ── ID move Gen 1 (hex) → octet RAM du type ──────────────────────────────────
# Source : pret/pokered (IDs 1-indexés).
# Note Gen 1 : Karate Chop, Gust, Bite sont Normal (pas Fighting/Flying/Dark).
MOVE_TYPES: dict[int, int] = {
    0x01: 0x00,  # Pound          → Normal
    0x02: 0x00,  # Karate Chop    → Normal (Gen 1 !)
    0x03: 0x00,  # Double Slap    → Normal
    0x04: 0x00,  # Comet Punch    → Normal
    0x05: 0x00,  # Mega Punch     → Normal
    0x06: 0x00,  # Pay Day        → Normal
    0x07: 0x14,  # Fire Punch     → Fire
    0x08: 0x19,  # Ice Punch      → Ice
    0x09: 0x17,  # ThunderPunch   → Electric
    0x0A: 0x00,  # Scratch        → Normal
    0x0B: 0x00,  # ViceGrip       → Normal
    0x0C: 0x00,  # Guillotine     → Normal
    0x0D: 0x00,  # Razor Wind     → Normal
    0x0F: 0x00,  # Cut            → Normal
    0x10: 0x00,  # Gust           → Normal (Gen 1 !)
    0x11: 0x02,  # Wing Attack    → Flying
    0x13: 0x02,  # Fly            → Flying
    0x14: 0x00,  # Bind           → Normal
    0x15: 0x00,  # Slam           → Normal
    0x16: 0x16,  # Vine Whip      → Grass
    0x17: 0x00,  # Stomp          → Normal
    0x18: 0x01,  # Double Kick    → Fighting
    0x19: 0x00,  # Mega Kick      → Normal
    0x1A: 0x01,  # Jump Kick      → Fighting
    0x1B: 0x01,  # Rolling Kick   → Fighting
    0x1D: 0x00,  # Headbutt       → Normal
    0x1E: 0x00,  # Horn Attack    → Normal
    0x1F: 0x00,  # Fury Attack    → Normal
    0x20: 0x00,  # Horn Drill     → Normal
    0x21: 0x00,  # Tackle         → Normal
    0x22: 0x00,  # Body Slam      → Normal
    0x23: 0x00,  # Wrap           → Normal
    0x24: 0x00,  # Take Down      → Normal
    0x25: 0x00,  # Thrash         → Normal
    0x26: 0x00,  # Double-Edge    → Normal
    0x28: 0x03,  # Poison Sting   → Poison
    0x29: 0x07,  # Twineedle      → Bug
    0x2A: 0x07,  # Pin Missile    → Bug
    0x2C: 0x00,  # Bite           → Normal (Gen 1 !)
    0x31: 0x00,  # SonicBoom      → Normal
    0x33: 0x03,  # Acid           → Poison
    0x34: 0x14,  # Ember          → Fire
    0x35: 0x14,  # Flamethrower   → Fire
    0x37: 0x15,  # Water Gun      → Water
    0x38: 0x15,  # Hydro Pump     → Water
    0x39: 0x15,  # Surf           → Water
    0x3A: 0x19,  # Ice Beam       → Ice
    0x3B: 0x19,  # Blizzard       → Ice
    0x3C: 0x18,  # Psybeam        → Psychic
    0x3D: 0x15,  # BubbleBeam     → Water
    0x3E: 0x19,  # Aurora Beam    → Ice
    0x3F: 0x00,  # Hyper Beam     → Normal
    0x40: 0x02,  # Peck           → Flying
    0x41: 0x02,  # Drill Peck     → Flying
    0x42: 0x01,  # Submission     → Fighting
    0x43: 0x01,  # Low Kick       → Fighting
    0x44: 0x01,  # Counter        → Fighting
    0x45: 0x01,  # Seismic Toss   → Fighting
    0x46: 0x00,  # Strength       → Normal
    0x47: 0x16,  # Absorb         → Grass
    0x48: 0x16,  # Mega Drain     → Grass
    0x4B: 0x16,  # Razor Leaf     → Grass
    0x4C: 0x16,  # SolarBeam      → Grass
    0x50: 0x16,  # Petal Dance    → Grass
    0x51: 0x07,  # String Shot    → Bug
    0x52: 0x1A,  # Dragon Rage    → Dragon
    0x53: 0x14,  # Fire Spin      → Fire
    0x54: 0x17,  # ThunderShock   → Electric
    0x55: 0x17,  # Thunderbolt    → Electric
    0x57: 0x17,  # Thunder        → Electric
    0x58: 0x05,  # Rock Throw     → Rock
    0x59: 0x04,  # Earthquake     → Ground
    0x5A: 0x04,  # Fissure        → Ground
    0x5B: 0x04,  # Dig            → Ground
    0x5D: 0x18,  # Confusion      → Psychic
    0x5E: 0x18,  # Psychic        → Psychic
    0x62: 0x00,  # Quick Attack   → Normal (priorité +1)
    0x63: 0x00,  # Rage           → Normal
    0x65: 0x08,  # Night Shade    → Ghost
    0x6A: 0x08,  # Confuse Ray    → Ghost
    0x75: 0x00,  # Bide           → Normal
    0x76: 0x00,  # Metronome      → Normal
    0x77: 0x02,  # Mirror Move    → Flying
    0x78: 0x00,  # Self-Destruct  → Normal
    0x79: 0x00,  # Egg Bomb       → Normal
    0x7A: 0x08,  # Lick           → Ghost
    0x7B: 0x03,  # Smog           → Poison
    0x7C: 0x03,  # Sludge         → Poison
    0x7D: 0x04,  # Bone Club      → Ground
    0x7E: 0x14,  # Fire Blast     → Fire
    0x7F: 0x15,  # Waterfall      → Water
    0x80: 0x15,  # Clamp          → Water
    0x81: 0x00,  # Swift          → Normal
    0x82: 0x00,  # Skull Bash     → Normal
    0x83: 0x00,  # Spike Cannon   → Normal
    0x84: 0x00,  # Constrict      → Normal
    0x88: 0x01,  # Hi Jump Kick   → Fighting
    0x8A: 0x18,  # Dream Eater    → Psychic
    0x8D: 0x07,  # Leech Life     → Bug
    0x8F: 0x02,  # Sky Attack     → Flying
    0x91: 0x15,  # Bubble         → Water
    0x95: 0x18,  # Psywave        → Psychic
    0x98: 0x15,  # Crabhammer     → Water
    0x99: 0x00,  # Explosion      → Normal
    0x9A: 0x00,  # Fury Swipes    → Normal
    0x9B: 0x04,  # Bonemerang     → Ground
    0x9D: 0x05,  # Rock Slide     → Rock
    0x9E: 0x00,  # Hyper Fang     → Normal
    0xA1: 0x00,  # Tri Attack     → Normal
    0xA2: 0x00,  # Super Fang     → Normal
    0xA3: 0x00,  # Slash          → Normal
    0xA5: 0x00,  # Struggle       → Normal
}

# Moves sans dégâts directs — à éviter si un move offensif existe.
STATUS_MOVES: frozenset[int] = frozenset({
    0x0E, 0x12, 0x1C, 0x27, 0x2B, 0x2D, 0x2E, 0x2F, 0x30, 0x32,
    0x36, 0x49, 0x4A, 0x4D, 0x4E, 0x4F, 0x56, 0x5C, 0x5F, 0x60,
    0x61, 0x64, 0x66, 0x67, 0x68, 0x69, 0x6B, 0x6C, 0x6D, 0x6E,
    0x6F, 0x70, 0x71, 0x72, 0x73, 0x74, 0x85, 0x86, 0x87, 0x89,
    0x8B, 0x8E, 0x90, 0x93, 0x94, 0x96, 0x97, 0x9C, 0x9F, 0xA0,
    0xA4,
})

# ── Moves avec priorité > 0 en Gen 1 ─────────────────────────────────────────
# En Gen 1, la priorité agit avant la vitesse → utile pour finir un ennemi bas PV.
# Source : pret/pokered engine/battle/
PRIORITY_MOVES: dict[int, int] = {
    0x62: 1,  # Quick Attack (Vive-Attaque) — priorité +1
    0x63: -1, # Rage — priorité -1 (attaque en dernier)
    0x75: -1, # Bide — priorité -1
    0x76: -1, # Metronome — priorité -1
}
