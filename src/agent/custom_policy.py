"""
custom_policy.py — Architecture Actor-Critic CNN + GRU pour Pokémon Bleu.

Architecture complète du pipeline :

    Dict obs
    ├─ screen (3, 72, 80)      → NatureCNN (3 couches conv)    → embedding 512
    ├─ visited_mask (1, 48, 48)→ LightCNN  (2 couches conv)    → embedding 256
    └─ ram (16,)               → MLP 256×2                     → embedding 256
                                                           concat → E_t (1024)
                                                    GRU(1024, 512) → h_t (512)
                               ┌───────────────────────────────────┤
                        Actor head                          Critic head
                    Linear(512, 7)                         Linear(512, 1)
                       + Softmax                             scalaire


Notes d'intégration SB3 :
  - PokemonFeaturesExtractor est compatible avec MaskablePPO
    (BaseFeaturesExtractor standard → plug-and-play).
  - PokemonGRUPolicy sous-classe MaskableActorCriticPolicy :
      • Rollout (collect_rollouts) : hidden state maintenu pas-à-pas ✓
      • Training (evaluate_actions) : hidden state ré-initialisé par
        mini-batch — BPTT approximatif (limitation connue de SB3 PPO).
        Un vrai BPTT complet nécessiterait RecurrentPPO ou PufferLib.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


# ── Dimensions cibles ─────────────────────────────────────────────────────────
_SCREEN_EMB  = 512   # sortie du CNN écran
_MASK_EMB    = 256   # sortie du CNN masque de visite
_RAM_EMB     = 256   # sortie du MLP RAM
_FUSION_DIM  = _SCREEN_EMB + _MASK_EMB + _RAM_EMB   # = 1024
GRU_HIDDEN   = 512   # taille de l'état caché GRU


# ── Blocs convolutifs ─────────────────────────────────────────────────────────

class NatureCNN(nn.Module):
    """CNN 3 couches inspiré de l'architecture Nature DQN (DeepMind).

    Conv1 : 32 filtres 8×8, stride 4 → extraction des structures macroscopiques.
    Conv2 : 64 filtres 4×4, stride 2 → motifs intermédiaires.
    Conv3 : 64 filtres 3×3, stride 1 → détails fins.
    Linear : projection vers output_dim.

    Args:
        in_channels : nombre de canaux d'entrée (3 pour l'écran empilé).
        input_h     : hauteur de l'image d'entrée.
        input_w     : largeur de l'image d'entrée.
        output_dim  : dimension de l'embedding de sortie.
    """

    def __init__(
        self,
        in_channels: int,
        input_h: int,
        input_w: int,
        output_dim: int = _SCREEN_EMB,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),          nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),          nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out = self._cnn_output_dim(in_channels, input_h, input_w)
        self.linear = nn.Sequential(nn.Linear(cnn_out, output_dim), nn.ReLU())

    def _cnn_output_dim(self, c: int, h: int, w: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            return int(self.cnn(dummy).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))


class LightCNN(nn.Module):
    """CNN 2 couches allégé pour le masque de visite binaire (1, 48, 48).

    Conv1 : 16 filtres 4×4, stride 2.
    Conv2 : 32 filtres 3×3, stride 2.
    Linear : projection vers output_dim.

    Args:
        in_channels : 1 (masque binaire).
        input_h     : 48.
        input_w     : 48.
        output_dim  : dimension de l'embedding de sortie.
    """

    def __init__(
        self,
        in_channels: int,
        input_h: int,
        input_w: int,
        output_dim: int = _MASK_EMB,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),          nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out = self._cnn_output_dim(in_channels, input_h, input_w)
        self.linear = nn.Sequential(nn.Linear(cnn_out, output_dim), nn.ReLU())

    def _cnn_output_dim(self, c: int, h: int, w: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            return int(self.cnn(dummy).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))


# ── Extracteur de caractéristiques ────────────────────────────────────────────

class PokemonFeaturesExtractor(BaseFeaturesExtractor):
    """Fusionne les trois modalités de l'espace Dict en un vecteur E_t.

    Entrée : Dict { 'screen', 'visited_mask', 'ram' }
    Sortie : tenseur E_t de dimension _FUSION_DIM (1024).

    Compatible avec MaskablePPO comme BaseFeaturesExtractor standard.
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=_FUSION_DIM)

        screen_shape = observation_space['screen'].shape        # (3, 72, 80)
        mask_shape   = observation_space['visited_mask'].shape  # (1, 48, 48)
        ram_dim      = observation_space['ram'].shape[0]        # 16

        self.screen_cnn = NatureCNN(
            in_channels=screen_shape[0],
            input_h=screen_shape[1],
            input_w=screen_shape[2],
            output_dim=_SCREEN_EMB,
        )

        self.mask_cnn = LightCNN(
            in_channels=mask_shape[0],
            input_h=mask_shape[1],
            input_w=mask_shape[2],
            output_dim=_MASK_EMB,
        )

        self.ram_mlp = nn.Sequential(
            nn.Linear(ram_dim, 256), nn.ReLU(),
            nn.Linear(256, _RAM_EMB), nn.ReLU(),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        screen_emb = self.screen_cnn(obs['screen'])
        mask_emb   = self.mask_cnn(obs['visited_mask'])
        ram_emb    = self.ram_mlp(obs['ram'])
        return torch.cat([screen_emb, mask_emb, ram_emb], dim=-1)


# ── Politique Actor-Critic avec GRU ───────────────────────────────────────────

class PokemonGRUPolicy(MaskableActorCriticPolicy):
    """Politique Actor-Critic CNN + GRU pour MaskablePPO.

    Architecture :
      E_t (1024) → GRU(hidden=512) → h_t
      h_t → Actor head  Linear(512, n_actions)  [7 actions]
      h_t → Critic head Linear(512, 1)

    Gestion de l'état caché :
      - Pendant le rollout (collect_rollouts) : _gru_hidden est mis à jour
        pas-à-pas pour chaque env. Les fins d'épisodes (done=True) réinitialisent
        les hidden states des envs concernés.
      - Pendant l'entraînement (evaluate_actions) : les hidden states sont
        ré-initialisés à zéro par mini-batch (BPTT approximatif).
        Limitation connue : pas de propagation des gradients au-delà du batch.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        **kwargs,
    ):
        # Injecte notre extracteur et désactive le mlp_extractor par défaut
        kwargs['features_extractor_class'] = PokemonFeaturesExtractor
        kwargs['features_extractor_kwargs'] = {}
        # net_arch vide : on gère nous-mêmes les couches intermédiaires via GRU
        kwargs.setdefault('net_arch', [])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

        # ── GRU ───────────────────────────────────────────────────────────────
        # Remplace le mlp_extractor de SB3 (qui ne ferait rien avec net_arch=[])
        self.gru = nn.GRU(
            input_size=_FUSION_DIM,
            hidden_size=GRU_HIDDEN,
            num_layers=1,
            batch_first=True,
        )

        # ── Têtes Actor-Critic ────────────────────────────────────────────────
        n_actions = action_space.n
        self.action_net = nn.Linear(GRU_HIDDEN, n_actions)
        self.value_net  = nn.Linear(GRU_HIDDEN, 1)

        # ── État caché GRU — buffer interne (1, n_envs, GRU_HIDDEN) ──────────
        # Initialisé lors du premier appel à _reset_hidden().
        self._gru_hidden: Optional[torch.Tensor] = None
        self._n_envs: int = 1

        # Ré-initialise tous les poids (action_net/value_net écrasent ceux de SB3)
        self._init_weights()

    # ── Initialisation des poids ──────────────────────────────────────────────

    def _init_weights(self):
        """Initialisation orthogonale standard pour les couches linéaires."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)

    # ── Gestion de l'état caché ───────────────────────────────────────────────

    def _reset_hidden(self, n_envs: int):
        """Alloue ou réinitialise complètement le buffer d'état caché."""
        self._n_envs = n_envs
        self._gru_hidden = torch.zeros(
            1, n_envs, GRU_HIDDEN, device=self.device
        )

    def reset_hidden_for_envs(self, env_indices: List[int]):
        """Réinitialise le hidden state pour les envs dont l'épisode vient de finir."""
        if self._gru_hidden is None:
            return
        for idx in env_indices:
            self._gru_hidden[0, idx, :] = 0.0

    # ── Forward passes ────────────────────────────────────────────────────────

    def _gru_step(
        self,
        features: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Un pas GRU sur un batch de features.

        Args:
            features : (batch, _FUSION_DIM) ou (batch, seq, _FUSION_DIM)
            hidden   : (1, batch, GRU_HIDDEN) ou None

        Returns:
            out    : (batch, GRU_HIDDEN)
            hidden : (1, batch, GRU_HIDDEN) mis à jour
        """
        if features.dim() == 2:
            features = features.unsqueeze(1)   # (batch, 1, fusion_dim)
        out, hidden = self.gru(features, hidden)
        return out.squeeze(1), hidden           # (batch, gru_hidden)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward complet : obs → (actions, values, log_probs).

        Utilisé pendant collect_rollouts (pas-à-pas par env).
        Met à jour _gru_hidden en place.
        """
        features = self.extract_features(obs)

        # Initialisation lazy du buffer (n_envs déduit de la taille du batch)
        batch_size = features.shape[0]
        if self._gru_hidden is None or self._gru_hidden.shape[1] != batch_size:
            self._reset_hidden(batch_size)

        gru_out, self._gru_hidden = self._gru_step(features, self._gru_hidden)
        self._gru_hidden = self._gru_hidden.detach()  # stop gradient pour rollout

        distribution = self._get_action_dist_from_latent(gru_out)
        if action_masks is not None:
            distribution.apply_masking(action_masks)

        actions  = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values   = self.value_net(gru_out)

        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Évalue un batch de transitions pour le calcul de la perte PPO.

        Pendant l'entraînement, le hidden state est ré-initialisé à zéro
        au début de chaque mini-batch (BPTT approximatif).
        """
        features = self.extract_features(obs)
        batch_size = features.shape[0]

        # Hidden state à zéro pour l'entraînement (BPTT tronqué)
        hidden = torch.zeros(1, batch_size, GRU_HIDDEN, device=self.device)
        gru_out, _ = self._gru_step(features, hidden)

        distribution = self._get_action_dist_from_latent(gru_out)
        if action_masks is not None:
            distribution.apply_masking(action_masks)

        log_prob = distribution.log_prob(actions)
        entropy  = distribution.entropy()
        values   = self.value_net(gru_out)

        return values, log_prob, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Prédit les valeurs d'état (tête Critique uniquement)."""
        features = self.extract_features(obs)
        batch_size = features.shape[0]

        if self._gru_hidden is None or self._gru_hidden.shape[1] != batch_size:
            hidden = torch.zeros(1, batch_size, GRU_HIDDEN, device=self.device)
        else:
            hidden = self._gru_hidden

        gru_out, _ = self._gru_step(features, hidden)
        return self.value_net(gru_out)

    def _predict(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Override : force le passage par forward() (GRU) au lieu du chemin SB3 par défaut.

        Le chemin SB3 standard (_predict → get_distribution → mlp_extractor)
        bypasse le GRU et envoie 1024 dims à action_net(512, 7) → RuntimeError.
        """
        actions, _, _ = self.forward(observation, deterministic=deterministic, action_masks=action_masks)
        return actions

    def _get_action_dist_from_latent(self, latent: torch.Tensor):
        """Construit la distribution d'actions depuis le vecteur latent."""
        mean_actions = self.action_net(latent)
        return self.action_dist.proba_distribution(action_logits=mean_actions)
