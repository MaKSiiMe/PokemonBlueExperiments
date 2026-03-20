"""
waypoints.py — Curriculum Bourg Palette → Brock, source unique de vérité.

Format : (map_id, x, y, state_file, label, max_steps)
  - map_id    : ID de la map cible (RAM 0xD35E)
  - x, y      : coordonnées cibles sur cette map
  - state_file: .state de départ pour entraîner ce segment
  - label     : description lisible pour les logs
  - max_steps : budget actions/épisode (phase fine-tune)
"""

WAYPOINTS: list[tuple[int, int, int, str, str, int]] = [
    # ── Bourg Palette ─────────────────────────────────────────────────────────
    # 00_pallet_town.state : map=0x00  spawn=(12,12)  post-tuto, starter+colis livré
    (0x00, 11,  1, 'states/00_pallet_town.state',                      "Bourg Palette → Route 1",        200),

    # ── Route 1 ───────────────────────────────────────────────────────────────
    (0x0C,  7, 26, 'states/07_route1_grass.state',            "Route 1 → checkpoint PNJ",       150),
    (0x0C,  7,  7, 'states/08_route1_pnj1.state',             "Route 1 → zone corniches",       200),
    (0x0C,  7,  1, 'states/09_route1_ledges.state',           "Route 1 → Jadielle City",        100),

    # ── Jadielle City ─────────────────────────────────────────────────────────
    (0x01, 19,  1, 'states/14_viridian_up.state',             "Jadielle → Route 2 nord",        200),

    # ── Route 2 ───────────────────────────────────────────────────────────────
    (0x0D,  6,  1, 'states/22_route2_down.state',             "Route 2 → Forêt de Jade",       1500),

    # ── Forêt de Jade (map 0x33) ──────────────────────────────────────────────
    (0x33, 26, 41, 'states/24_viridian_forest_down.state',    "Forêt → zone 1",                 200),
    (0x33, 26, 25, 'states/26_viridian_forest_battle1.state', "Forêt → zone 2",                 200),
    (0x33, 26,  9, 'states/27_viridian_forest_grass1.state',  "Forêt → zone 4",                 200),
    (0x33,  6, 15, 'states/29_viridian_forest_item2.state',   "Forêt → zone 5",                 300),
    (0x33,  7, 25, 'states/30_viridian_forest_grass2.state',  "Forêt → zone 6",                 150),
    (0x33,  1, 16, 'states/31_viridian_forest_item3.state',   "Forêt → zone 7",                 150),
    (0x33,  1,  1, 'states/32_viridian_forest_battle3.state', "Forêt → sortie nord",            150),

    # ── Argenta City + Arène ──────────────────────────────────────────────────
    (0x02, 16, 16, 'states/35_pewter_center_front.state',     "Argenta → Arène",                300),
    (0x36,  5,  5, 'states/47_pewter_gym.state',              "Arène → Pierre (Brock)",         200),
]
