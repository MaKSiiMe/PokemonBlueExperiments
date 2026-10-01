# Data Processing — Pokémon Blue RL Agent

## Data Sources

The project uses a single data source: **real-time RAM reads from the Pokémon Blue ROM running inside the PyBoy emulator**. There is no aggregation from external datasets, APIs, or web scraping.

At each decision step, the agent reads 16 specific memory addresses from the Game Boy's RAM (see [ram_map.py](../src/emulator/ram_map.py) for the complete address reference). These addresses expose game state such as player position, HP, active map, battle status, event progression flags, and party data.

Additionally, **curriculum save states** (`.state` files in `states/`) serve as structured starting points for training. These are manually captured snapshots of game progress at 17 key waypoints between Pallet Town and Pewter Gym. They are not training data per se, but they define where each training episode begins.

**Sources summary:**
| Source | What it provides |
|---|---|
| PyBoy RAM reads | All observations (position, HP, battle state, flags, etc.) |
| PyBoy screen buffer | Raw pixel frames for the CNN branch |
| `.state` save files | Curriculum episode starting points |
| `mapping.json` | Zone connectivity and encounter rates (used at load time) |

---

## Data Format

**Raw format:** unsigned integers read directly from RAM — either `uint8` (single byte) or `uint16` (two bytes, big-endian for HP values). The screen is a raw Game Boy LCD buffer (160×144 pixels, 4-shade grayscale).

**Transformed format:** all inputs are converted to `float32` tensors before reaching the neural network:

| Observation key | Raw format | Transformed shape | Range |
|---|---|---|---|
| `ram` | ~16 uint8/uint16 values | `(16,)` float32 | `[0, 1]` |
| `screen` | 160×144 uint8 pixels | `(3, 72, 80)` float32 | `[0, 1]` |
| `visited_mask` | binary tile grid | `(1, 48, 48)` float32 | `{0, 1}` |

The observation dictionary is the final format consumed by the PPO policy (defined in [custom_policy.py](../src/agent/custom_policy.py)).

---

## Features Currently in the Data

The 16-element `ram` vector contains:

| Index | Feature | RAM address | Normalization |
|---|---|---|---|
| 0 | `player_x` | `0xD362` | ÷ 255 |
| 1 | `player_y` | `0xD361` | ÷ 255 |
| 2 | `map_id` | `0xD35E` | ÷ 255 |
| 3 | `direction` | `0xD35D` | mapped to {0, 0.33, 0.66, 1} |
| 4 | `hp_pct` | `0xD16C–D16D` / `0xD18C–D18E` | current ÷ max |
| 5 | `battle_status` | `0xD057` | ÷ 2 |
| 6 | `event_flags_pct` | `0xD747` block | flags set ÷ total |
| 7 | `steps_stuck_norm` | computed | ÷ 100, clamped to [0, 1] |
| 8 | `badges_pct` | `0xD356` | badges earned ÷ 8 |
| 9 | `type_advantage` | computed from knowledge graph | best multiplier ÷ 4.0 |
| 10 | `enemy_can_evolve` | computed | binary {0, 1} |
| 11 | `zone_density` | `mapping.json` lookup | encounter rate ÷ 8.0 |
| 12 | `battle_mon_hp_pct` | party RAM block | current ÷ max |
| 13 | `pokedex_pct` | `0xD2F7` bitmask | caught ÷ 151 |
| 14 | `money_norm` | `0xD347–D349` BCD | decoded ÷ 999999 |
| 15 | `items_norm` | `0xCF7B` count | ÷ 20 |

**Data exploration planned:**
- Distribution of each RAM feature across episodes and zones (histograms)
- Correlation matrix between features and episode reward
- Visited map coverage heatmaps to assess exploration quality
- Reward signal breakdown per episode step (TensorBoard)
- Feature importance analysis post-training (permutation importance on RAM vector)

---

## Hypotheses About the Data

**Hypothesis 1 — HP % is the strongest signal for survival decisions.**  
The agent should learn to heal or retreat when `hp_pct` falls below ~0.2. Testable by inspecting the policy's action distribution conditioned on `hp_pct` buckets.

**Hypothesis 2 — Type advantage dominates battle feature importance.**  
The `type_advantage` feature (derived from the Gen 1 type chart via [graph.py](../src/knowledge/graph.py)) should have the highest weight in the battle branch. Testable by ablation: train without this feature and measure win rate against Brock.

**Hypothesis 3 — Zone density correlates with navigation difficulty.**  
High `zone_density` zones (tall grass, caves) slow exploration because wild encounters interrupt navigation. Testable by plotting average steps-per-tile vs. zone density across training runs.

**Testing approach:** After training reaches a stable policy, freeze the model and run systematic ablations — zeroing out individual RAM features one at a time — and measure the drop in cumulative reward. Features causing the largest drop are considered most influential.

---

## Sparsity, Missing Data, and Outliers

**Missing data:** None possible — PyBoy always returns a value for every RAM address. The emulator is deterministic and the ROM is complete. There is no concept of a "null" game state.

**Sparsity:** The `visited_mask` grid starts as all zeros at the beginning of each episode and fills in as the agent explores. Early episodes are highly sparse in this dimension. This is expected and intentional — it encodes novel vs. revisited territory.

**Outliers / degenerate states:**
- **Stuck agent:** if the player does not move for too many consecutive steps, `steps_stuck_norm` rises and a penalty is applied. This handles the degenerate case of the agent pressing into a wall repeatedly.
- **Death:** handled by episode termination + reward penalty. The episode resets to the curriculum save state.
- **Infinite dialog loops:** the orchestrator auto-presses A when `0xD11C != 0` (text active flag), preventing the agent from getting permanently stuck in a dialog box.

---

## Train / Validation / Test Split

This is an RL setting, so the split is temporal and curriculum-based rather than a random partition of a static dataset:

| Split | Definition | Purpose |
|---|---|---|
| **Train** | PPO rollouts collected online (2048 steps × 16 parallel envs per update) | Policy gradient learning |
| **Validation** | Held-out save states at each of the 17 curriculum waypoints | Measure waypoint-reaching success rate without curriculum assist |
| **Test** | Full unassisted run from Pallet Town to Brock defeat (no save state jump) | Final end-to-end evaluation |

Checkpoints are saved every 50k training steps. The validation metric is whether the agent reaches the next waypoint within a fixed step budget when started from the previous one.

---

## Ensuring an Unbiased Dataset

**Label noise:** not applicable — the ROM is fully deterministic. The same RAM state always leads to the same next state for a given action. There is no labeling ambiguity.

**Curriculum bias:** training always from the same save states could cause the agent to overfit to specific starting conditions. This is mitigated by:
- Randomizing the number of frames advanced after loading a state (slight position jitter)
- Mixing episodes from adjacent waypoints during training

**Exploration bias:** without deliberate exploration incentives, the agent would only see states near its current policy's trajectory. The visited mask reward and entropy coefficient (`ent_coef=0.01`) in PPO push the agent to explore novel states.

**Representation bias:** the knowledge graph ([gen1_data.py](../src/knowledge/gen1_data.py)) encodes the actual Gen 1 type chart, move database, and encounter tables. These are ground-truth game mechanics — no bias is introduced here beyond the original game design.

---

## Features Included in Model Training

The PPO policy ([custom_policy.py](../src/agent/custom_policy.py)) receives all three observation components:

1. **`ram` (16 floats):** processed by a 2-layer MLP (256 units each) → 256-dim embedding
2. **`screen` (3 × 72 × 80):** processed by NatureCNN (3 conv layers) → 512-dim embedding
3. **`visited_mask` (1 × 48 × 48):** processed by LightCNN (2 conv layers) → 256-dim embedding

All three embeddings are concatenated (1024-dim) and passed through a GRU (hidden=512) for temporal memory. The actor and critic heads each read from the GRU output.

The battle heuristic agent ([battle_agent.py](../src/agent/battle_agent.py)) uses only: enemy type, active Pokémon movepool, enemy HP %, and the type chart lookup. It does not use the neural network.

---

## Data Types

| Feature group | Type | Notes |
|---|---|---|
| RAM vector (16 features) | Numerical (continuous float32) | All normalized to [0, 1] |
| Direction | Ordinal → numerical | Mapped to 4 float values |
| Battle status | Categorical → numerical | 3 classes mapped to {0, 0.5, 1} |
| Screen frames | Spatial (image tensor) | Grayscale, stacked |
| Visited mask | Spatial (binary image) | Tile-level binary grid |
| Type advantage | Numerical (discrete multipliers) | {0, 0.5, 1, 2, 4} → normalized |

There are no free-text or audio features. All inputs are numeric or image tensors.

---

## Data Transformations for the Model

| Raw value | Transformation | Result |
|---|---|---|
| RAM uint8 (most features) | divide by 255 | float32 in [0, 1] |
| HP (uint16, two bytes) | big-endian decode → current ÷ max | float32 in [0, 1] |
| Direction byte | lookup table | float32 in {0, 0.33, 0.66, 1} |
| Money (BCD, 3 bytes) | BCD decode → ÷ 999999 | float32 in [0, 1] |
| Screen pixels (uint8) | grayscale conversion, 2× downsample, ÷ 255, stack 3 frames | (3, 72, 80) float32 |
| Visited tiles | binary set → 2D grid | (1, 48, 48) float32 |
| Type advantage | type chart lookup → multiplier ÷ 4.0 | float32 in [0, 1] |
| Zone density | `mapping.json` lookup → encounter rate ÷ 8.0 | float32 in [0, 1] |

All transformations are applied inside `PokemonBlueEnv._get_obs()` ([pokemon_env.py](../src/emulator/pokemon_env.py)) before the observation is returned to the agent. No offline preprocessing step is needed — data is generated and transformed on-the-fly at training time.

---

## Data Storage

| Artifact | Location | Format | Notes |
|---|---|---|---|
| Curriculum save states | `states/*.state` | PyBoy binary | One per waypoint (17 total) |
| Trained PPO models | `models/rl_checkpoints/*.zip` | Stable Baselines3 zip | Saved every 50k steps |
| TensorBoard logs | `logs/exploration/` | TensorBoard event files | Reward, loss, entropy curves |
| Zone map | `mapping.json` | JSON | Zone connectivity, encounter rates |
| Knowledge graph | In-memory (NetworkX) | Not persisted | Rebuilt from `gen1_data.py` at startup |

No external database or cloud storage is used. All data fits on a local disk. The ROM file (`PokemonBlue.gb`) must be provided by the user and is not distributed with the repository.
