# Archive — documentation de la première version (objectif : Badge Roche)

Ces documents décrivent l'agent end-to-end initial (CNN + GRU sur pixels, PPO).
Ils sont conservés pour l'historique du projet, mais ne reflètent plus le code.

Principales raisons :

- **Adresses RAM fausses** : `ram_map.md` et `knowledge_graph.md` citent des adresses
  vérifiées fausses contre `pokeblue.sym` (types et espèce ennemis, orientation, sac,
  « dialogue actif », transitions). La source de vérité est désormais
  `src/pokeblue/state/ram_symbols.py`, généré par `scripts/gen_ram_symbols.py`.
- **Données Gen 1** : les tables de types et de moves venaient en partie de PokéAPI
  (valeurs modernes). Elles sont désormais générées depuis pret/pokered
  (`scripts/gen_gen1_data.py`).
- **Architecture** : `architecture.md`, `pipeline.md`, `data_processing.md` et
  `roadmap.md` décrivent des espaces d'observation et d'action incohérents entre eux
  (16 ou 9 flottants, `Discrete(7)` ou `Discrete(6)`) et une architecture remplacée par
  l'approche modulaire (état RAM + base de connaissance + skills spécialisés).

La nouvelle documentation sera écrite au fil des phases de la refonte.
