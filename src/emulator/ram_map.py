"""
ram_map.py — Alias historiques des adresses RAM, dérivés de `pokeblue.state.ram_symbols`.

Couche de compatibilité pour l'ancien code (env, agents, outils) : aucune adresse
n'est écrite ici, tout vient de la table de symboles générée depuis pret/pokered.
Le nouveau code doit importer `pokeblue.state.ram_symbols` directement ; ce module
disparaîtra avec la lecture d'état `GameState` (Phase 1).

Supprimés (adresses fausses, sans équivalent en un octet — voir la détection de mode
de la Phase 1) : RAM_FADING (lisait wPrize3), RAM_TEXT_ACTIVE (wCapturedMonSpecies),
RAM_MENU (wLinkState).
"""

from pokeblue.state import ram_symbols as sym

# ── Position joueur ───────────────────────────────────────────────────────────
RAM_PLAYER_X     = sym.W_X_COORD
RAM_PLAYER_Y     = sym.W_Y_COORD
RAM_MAP_ID       = sym.W_CUR_MAP
# Orientation du sprite joueur : SPRITE_FACING_DOWN/UP/LEFT/RIGHT (0x0/0x4/0x8/0xC).
RAM_DIRECTION    = sym.W_SPRITE_PLAYER_STATE_DATA1_FACING_DIRECTION

# ── Combat — état ─────────────────────────────────────────────────────────────
RAM_BATTLE       = sym.W_IS_IN_BATTLE   # 0 = hors combat, WILD_BATTLE, TRAINER_BATTLE, LOST_BATTLE

# Pokémon joueur actif en combat — structure wBattleMon
RAM_BATTLE_MON_HP_H     = sym.W_BATTLE_MON_HP
RAM_BATTLE_MON_HP_L     = sym.W_BATTLE_MON_HP + 1
RAM_BATTLE_MON_MAX_HP_H = sym.W_BATTLE_MON_MAX_HP
RAM_BATTLE_MON_MAX_HP_L = sym.W_BATTLE_MON_MAX_HP + 1
RAM_BATTLE_MON_STATUS   = sym.W_BATTLE_MON_STATUS   # bits 0-2 = SLP, PSN/BRN/FRZ/PAR = bits 3/4/5/6
RAM_PLAYER_STATUS       = RAM_BATTLE_MON_STATUS

# Moves et PP du Pokémon *actif en combat* (wBattleMonMoves / wBattleMonPP).
# Un octet de PP contient aussi les PP Plus utilisés : masquer avec RAM_PP_MASK.
RAM_MOVE_IDS = tuple(sym.W_BATTLE_MON_MOVES + i for i in range(sym.NUM_MOVES))
RAM_MOVE_PP  = tuple(sym.W_BATTLE_MON_PP + i for i in range(sym.NUM_MOVES))
RAM_PP_MASK  = sym.PP_MASK

# Pokémon ennemi en combat — structure wEnemyMon
RAM_ENEMY_SPECIES = sym.W_ENEMY_MON_SPECIES   # ID interne Gen 1
RAM_ENEMY_LEVEL   = sym.W_ENEMY_MON_LEVEL
RAM_ENEMY_HP_H    = sym.W_ENEMY_MON_HP
RAM_ENEMY_HP_L    = sym.W_ENEMY_MON_HP + 1
RAM_ENEMY_STATUS  = sym.W_ENEMY_MON_STATUS
RAM_ENEMY_TYPE1   = sym.W_ENEMY_MON_TYPE1
RAM_ENEMY_TYPE2   = sym.W_ENEMY_MON_TYPE2
RAM_ENEMY_MHP_H   = sym.W_ENEMY_MON_MAX_HP
RAM_ENEMY_MHP_L   = sym.W_ENEMY_MON_MAX_HP + 1

# ── PV du premier Pokémon de l'équipe (hors combat) ───────────────────────────
RAM_PLAYER_HP_H  = sym.W_PARTY_MON1_HP
RAM_PLAYER_HP_L  = sym.W_PARTY_MON1_HP + 1
RAM_PLAYER_MHP_H = sym.W_PARTY_MON1_MAX_HP
RAM_PLAYER_MHP_L = sym.W_PARTY_MON1_MAX_HP + 1

# ── Équipe (wPartyMon1..6, valeurs 16 bits en big-endian) ─────────────────────
RAM_PARTY_COUNT = sym.W_PARTY_COUNT
_PARTY = range(1, sym.PARTY_LENGTH + 1)
RAM_PARTY_SPECIES = tuple(getattr(sym, f"W_PARTY_MON{i}_SPECIES") for i in _PARTY)
RAM_PARTY_LEVELS  = tuple(getattr(sym, f"W_PARTY_MON{i}_LEVEL") for i in _PARTY)
RAM_PARTY_HP      = tuple(getattr(sym, f"W_PARTY_MON{i}_HP") for i in _PARTY)       # octet fort
RAM_PARTY_MAX_HP  = tuple(getattr(sym, f"W_PARTY_MON{i}_MAX_HP") for i in _PARTY)   # octet fort

# ── Sac : wNumBagItems puis paires (item_id, quantité), BAG_ITEM_CAPACITY max ──
RAM_ITEM_COUNT = sym.W_NUM_BAG_ITEMS
RAM_ITEM_DATA  = sym.W_BAG_ITEMS
RAM_ITEM_CAPACITY = sym.BAG_ITEM_CAPACITY

# ── Argent : 3 octets BCD (octet fort en premier) ─────────────────────────────
RAM_MONEY = tuple(sym.W_PLAYER_MONEY + i for i in range(3))

# ── Pokédex : tableaux de bits, 1 bit par espèce ──────────────────────────────
RAM_POKEDEX_OWNED = sym.W_POKEDEX_OWNED
RAM_POKEDEX_SEEN  = sym.W_POKEDEX_SEEN
RAM_POKEDEX_LEN   = sym.POKEDEX_FLAGS_SIZE
RAM_POKEDEX_MAX   = sym.NUM_POKEMON

# ── Progression ───────────────────────────────────────────────────────────────
RAM_BADGES      = sym.W_OBTAINED_BADGES   # bit 0 = Badge Roche (BIT_BOULDERBADGE)
# wEventFlags : NUM_EVENTS = 2560 drapeaux sur EVENT_FLAGS_SIZE = 320 octets.
RAM_EVENT_FLAGS = sym.W_EVENT_FLAGS
RAM_EVENT_LEN   = sym.EVENT_FLAGS_SIZE

# ── Maps / warps ──────────────────────────────────────────────────────────────
RAM_WARP_COUNT = sym.W_NUMBER_OF_WARPS
RAM_WARP_DATA  = sym.W_WARP_ENTRIES
RAM_SIGN_COUNT = sym.W_NUM_SIGNS
RAM_SIGN_DATA  = sym.W_SIGN_COORDS
