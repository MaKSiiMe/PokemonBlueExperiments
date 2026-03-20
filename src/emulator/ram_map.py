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

# ── Combat ────────────────────────────────────────────────────────────────────
RAM_BATTLE       = 0xD057   # 0=overworld  1=sauvage  2=dresseur
RAM_ENEMY_LVL    = 0xD018
RAM_ENEMY_TYPE1  = 0xD01F
RAM_ENEMY_TYPE2  = 0xD020
RAM_MOVE_IDS     = (0xD173, 0xD174, 0xD175, 0xD176)
RAM_MOVE_PP      = (0xD188, 0xD189, 0xD18A, 0xD18B)

# ── HP joueur ─────────────────────────────────────────────────────────────────
RAM_PLAYER_HP_H  = 0xD16C
RAM_PLAYER_HP_L  = 0xD16D
RAM_PLAYER_MHP_H = 0xD18C
RAM_PLAYER_MHP_L = 0xD18D

# ── Interface / transitions ───────────────────────────────────────────────────
RAM_FADING       = 0xD13F   # transition d'écran (non-zero = en cours)
RAM_TEXT_ACTIVE  = 0xD11C   # boite de dialogue (non-zero)
RAM_MENU         = 0xD12B   # menu ouvert (non-zero)

# ── Progression ───────────────────────────────────────────────────────────────
RAM_BADGES       = 0xD356   # bitmask badges (bit 0 = Pierre/Brock, ...)
RAM_EVENT_FLAGS  = 0xD747   # 32 octets = 256 flags (dresseurs battus, items, etc.)
RAM_EVENT_LEN    = 32

# ── Maps / warps ──────────────────────────────────────────────────────────────
RAM_WARP_COUNT   = 0xD3AE
RAM_WARP_DATA    = 0xD3AF
RAM_SIGN_COUNT   = 0xD4B0
RAM_SIGN_DATA    = 0xD4B1
