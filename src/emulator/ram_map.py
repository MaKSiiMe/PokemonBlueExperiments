"""
ram_map.py — Adresses RAM Pokémon Bleu/Rouge (Gen 1, version US).
Source : pret/pokered

Source unique de vérité pour tous les modules du projet.
"""

# ── Position joueur ───────────────────────────────────────────────────────────
RAM_PLAYER_X     = 0xD362
RAM_PLAYER_Y     = 0xD361
RAM_MAP_ID       = 0xD35E
RAM_DIRECTION    = 0xD35D   # 0x00=bas  0x04=haut  0x08=gauche  0x0C=droite

# ── Combat — état ─────────────────────────────────────────────────────────────
RAM_BATTLE            = 0xD057   # 0=overworld  1=sauvage  2=dresseur

# Pokémon joueur actif en combat — structure wBattleMon (D014)
# Ces adresses reflètent le Pokémon actuellement en jeu, pas les données de l'équipe.
RAM_BATTLE_MON_HP_H     = 0xD015   # HP courants du Pokémon actif (octet fort)
RAM_BATTLE_MON_HP_L     = 0xD016   # HP courants du Pokémon actif (octet faible)
RAM_BATTLE_MON_MAX_HP_H = 0xD023   # HP max du Pokémon actif (octet fort)
RAM_BATTLE_MON_MAX_HP_L = 0xD024   # HP max du Pokémon actif (octet faible)
RAM_BATTLE_MON_STATUS   = 0xD018   # altération d'état (masque de bits)
                                    # Bit 6=PAR  Bit 5=GEL  Bit 4=BRL  Bit 3=PSN  Bits 0-2=SLP

# Pokémon joueur (en combat) — moves
RAM_PLAYER_STATUS     = 0xD018   # alias pour RAM_BATTLE_MON_STATUS
RAM_MOVE_IDS          = (0xD173, 0xD174, 0xD175, 0xD176)
RAM_MOVE_PP           = (0xD188, 0xD189, 0xD18A, 0xD18B)

# Pokémon ennemi (en combat)
# Structure wEnemyMon commence à 0xCFDE (source : pret/pokered)
RAM_ENEMY_SPECIES     = 0xCFDE   # ID interne Gen 1 de l'espèce ennemie
RAM_ENEMY_LEVEL       = 0xCFE3   # niveau ennemi
RAM_ENEMY_HP_H        = 0xCFE7   # HP actuels ennemi (octet fort)
RAM_ENEMY_HP_L        = 0xCFE8   # HP actuels ennemi (octet faible)
RAM_ENEMY_STATUS      = 0xCFE9   # statut ennemi (PAR/SLP/PSN…)
RAM_ENEMY_TYPE1       = 0xD01F
RAM_ENEMY_TYPE2       = 0xD020
RAM_ENEMY_MHP_H       = 0xCFF4   # HP max ennemi (octet fort)
RAM_ENEMY_MHP_L       = 0xCFF5   # HP max ennemi (octet faible)

# ── HP joueur (données équipe, hors combat) ───────────────────────────────────
RAM_PLAYER_HP_H  = 0xD16C
RAM_PLAYER_HP_L  = 0xD16D
RAM_PLAYER_MHP_H = 0xD18D   # D18C = level byte; D18D-D18E = max HP (big-endian)
RAM_PLAYER_MHP_L = 0xD18E

# ── Équipe joueur (party) ──────────────────────────────────────────────────────
# Structure : wPartyMon1 = 0xD16B, taille par Pokémon = 0x2C (44 octets)
# Offsets dans la structure (source : pret/pokered) :
#   +0x00         : espèce (ID interne Gen 1)
#   +0x01 / +0x02 : HP courants (big-endian)
#   +0x21         : niveau
#   +0x22 / +0x23 : HP max (big-endian)
RAM_PARTY_COUNT  = 0xD163   # nombre de Pokémon dans l'équipe (0-6)

# Espèces — wPartyMonN + 0x00
RAM_PARTY_SPECIES = (
    0xD16B,   # Pokémon 1
    0xD197,   # Pokémon 2
    0xD1C3,   # Pokémon 3
    0xD1EF,   # Pokémon 4
    0xD21B,   # Pokémon 5
    0xD247,   # Pokémon 6
)

# Niveaux — wPartyMonN + 0x21
RAM_PARTY_LEVELS = (
    0xD18C,   # Pokémon 1 — D16B + 0x21
    0xD1B8,   # Pokémon 2 — D197 + 0x21
    0xD1E4,   # Pokémon 3 — D1C3 + 0x21
    0xD210,   # Pokémon 4 — D1EF + 0x21
    0xD23C,   # Pokémon 5 — D21B + 0x21
    0xD268,   # Pokémon 6 — D247 + 0x21
)

# HP courants (octet fort) — wPartyMonN + 0x01
RAM_PARTY_HP = (
    0xD16C,   # Pokémon 1 — D16B + 0x01
    0xD198,   # Pokémon 2 — D197 + 0x01
    0xD1C4,   # Pokémon 3 — D1C3 + 0x01
    0xD1F0,   # Pokémon 4 — D1EF + 0x01
    0xD21C,   # Pokémon 5 — D21B + 0x01
    0xD248,   # Pokémon 6 — D247 + 0x01
)

# HP maximum (octet fort) — wPartyMonN + 0x22
RAM_PARTY_MAX_HP = (
    0xD18D,   # Pokémon 1 — D16B + 0x22
    0xD1B9,   # Pokémon 2 — D197 + 0x22
    0xD1E5,   # Pokémon 3 — D1C3 + 0x22
    0xD211,   # Pokémon 4 — D1EF + 0x22
    0xD23D,   # Pokémon 5 — D21B + 0x22
    0xD269,   # Pokémon 6 — D247 + 0x22
)

# ── Inventaire ────────────────────────────────────────────────────────────────
# Structure wNumBagItems (CF7B) + paires (item_id, quantité) × 20 max
# CF7B : nombre d'objets uniques dans le sac (max 20)
# CF7C-CF7D : objet 1 (id, qté) … CF99-CF9A : objet 20 (id, qté)
RAM_ITEM_COUNT = 0xCF7B   # nombre d'entrées uniques dans le sac (0-20)
RAM_ITEM_DATA  = 0xCF7C   # début des paires (item_id, quantité)

# ── Argent (Pokédollars) ──────────────────────────────────────────────────────
# Encodé en Binary Coded Decimal (BCD) sur 3 octets.
# Décodage : chaque quartet = un chiffre décimal.
# Ex : 0x09 0x99 0x99 → 99 999 Pokédollars
# Max : 0x99 0x99 0x99 → 999 999 Pokédollars
RAM_MONEY = (0xD347, 0xD348, 0xD349)   # (octet fort, octet moyen, octet faible)

# ── Pokédex ───────────────────────────────────────────────────────────────────
# wPokedexOwned (D2F7) : 19 octets, masque de bits — bit à 1 = espèce possédée
# wPokedexSeen  (D30A) : 19 octets, masque de bits — bit à 1 = espèce aperçue
# Ensemble couvre les 151 Pokémon Gen 1 (152 bits, dernier bit inutilisé)
RAM_POKEDEX_OWNED     = 0xD2F7   # début masque "capturés"  (19 octets)
RAM_POKEDEX_SEEN      = 0xD30A   # début masque "aperçus"   (19 octets)
RAM_POKEDEX_LEN       = 19       # octets par masque
RAM_POKEDEX_MAX       = 151      # nombre total d'espèces Gen 1

# ── Interface / transitions ───────────────────────────────────────────────────
RAM_FADING       = 0xD13F   # transition d'écran (non-zero = en cours)
RAM_TEXT_ACTIVE  = 0xD11C   # boite de dialogue (non-zero)
RAM_MENU         = 0xD12B   # menu ouvert (non-zero)

# ── Progression ───────────────────────────────────────────────────────────────
RAM_BADGES       = 0xD356   # bitmask badges (bit 0 = Pierre/Brock, ...)

# Flags d'événements généraux — wEventFlags (D747), 0x140 octets = 320 drapeaux
# Couvre : dresseurs battus, objets uniques ramassés, PNJ activés, etc.
# D7BA est l'adresse du premier "milestone flag" significatif dans ce bloc
# (D7BA = D747 + 0x73), utilisé comme référence dans certaines documentations.
RAM_EVENT_FLAGS  = 0xD747   # base des event flags
RAM_EVENT_LEN    = 32       # octets lus pour le comptage (256 drapeaux)

# ── Maps / warps ──────────────────────────────────────────────────────────────
RAM_WARP_COUNT   = 0xD3AE
RAM_WARP_DATA    = 0xD3AF
RAM_SIGN_COUNT   = 0xD4B0
RAM_SIGN_DATA    = 0xD4B1
