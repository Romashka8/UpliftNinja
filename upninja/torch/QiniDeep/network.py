# ----------------------------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

# ----------------------------------------------------------------------------------------------------------------------------------------


OutcomeType = Literal["binary", "continuous"]
NormType = Literal["none", "layernorm", "batchnorm"]


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


def _make_norm(norm: NormType, dim: int) -> nn.Module:
    if norm == "none":
        return nn.Identity()
    if norm == "layernorm":
        return nn.LayerNorm(dim)
    if norm == "batchnorm":
        return nn.BatchNorm1d(dim)
    raise ValueError(f"Unknown norm: {norm}")


def build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    *,
    activation: str = "relu",
    dropout: float = 0.0,
    norm: NormType = "none",
    out_activation: Optional[str] = None,
) -> nn.Sequential:
    """
    Build an MLP:
      [Linear -> Norm -> Act -> Dropout] x len(hidden_dims) -> Linear(out_dim) -> (optional out_activation)

    hidden_dims are the *hidden* layers, out_dim is the final layer size.
    """
    layers: List[nn.Module] = []
    prev = in_dim
    act = _get_activation(activation)

    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(_make_norm(norm, h))
        layers.append(act)
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h

    layers.append(nn.Linear(prev, out_dim))
    if out_activation:
        layers.append(_get_activation(out_activation))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class DeepUpliftNetConfig:
    """
    QiniDeep / Deep Uplift Network (DUN) configuration.

    Control arm is always index 0.
    Treatment arms are indices 1..n_treatments.
    """
    n_features: int
    n_treatments: int = 1

    # Representation MLP (paper uses a simple ~2-hidden-layer setup; keep configurable)
    trunk_hidden_dims: Tuple[int, ...] = (50, 20)

    # Optional hidden layers inside each arm head (default: linear head)
    head_hidden_dims: Tuple[int, ...] = ()

    activation: str = "relu"
    dropout: float = 0.0
    norm: NormType = "none"

    outcome_type: OutcomeType = "binary"  # "binary" -> heads output logits; "continuous" -> raw values

    # "direct": predict Y_hat(w|x) for each arm directly (paper’s preferred implementation)
    # "residual": predict baseline Y0 and deltas per treatment: Y_hat(k)=Y0+delta_k
    parameterization: Literal["direct", "residual"] = "direct"

    # Optional ablation: Dragonnet-style propensity head
    use_propensity_head: bool = False
    propensity_loss_weight: float = 0.0  # if >0 and use_propensity_head=True, add CE loss to training


class DeepUpliftNetwork(nn.Module):
    """
    Deep Uplift Network (DUN): shared trunk + multiple output heads (one per arm).

    Forward returns dict:
      - "y_hat": (batch, n_arms)  [logits for binary outcome, raw for continuous]
      - optionally "t_logits": (batch, n_arms) if propensity head is enabled
    """
    def __init__(self, cfg: DeepUpliftNetConfig):
        super().__init__()
        self.cfg = cfg
        self.n_arms = 1 + cfg.n_treatments

        # --- trunk (shared representation) ---
        if len(cfg.trunk_hidden_dims) == 0:
            self.trunk = nn.Identity()
            trunk_out_dim = cfg.n_features
        else:
            trunk_out_dim = cfg.trunk_hidden_dims[-1]
            self.trunk = build_mlp(
                in_dim=cfg.n_features,
                hidden_dims=cfg.trunk_hidden_dims[:-1],
                out_dim=trunk_out_dim,
                activation=cfg.activation,
                dropout=cfg.dropout,
                norm=cfg.norm,
            )

        # --- outcome heads ---
        if cfg.parameterization == "direct":
            # Predict potential outcome for each arm directly.
            self.heads = nn.ModuleList([
                build_mlp(
                    in_dim=trunk_out_dim,
                    hidden_dims=cfg.head_hidden_dims,
                    out_dim=1,
                    activation=cfg.activation,
                    dropout=cfg.dropout,
                    norm=cfg.norm,
                )
                for _ in range(self.n_arms)
            ])
            self.y0_head = None
            self.delta_heads = None

        elif cfg.parameterization == "residual":
            # Predict baseline (control) + treatment deltas.
            self.y0_head = build_mlp(
                in_dim=trunk_out_dim,
                hidden_dims=cfg.head_hidden_dims,
                out_dim=1,
                activation=cfg.activation,
                dropout=cfg.dropout,
                norm=cfg.norm,
            )
            self.delta_heads = nn.ModuleList([
                build_mlp(
                    in_dim=trunk_out_dim,
                    hidden_dims=cfg.head_hidden_dims,
                    out_dim=1,
                    activation=cfg.activation,
                    dropout=cfg.dropout,
                    norm=cfg.norm,
                )
                for _ in range(cfg.n_treatments)
            ])
            self.heads = None
        else:
            raise ValueError(f"Unknown parameterization: {cfg.parameterization}")

        # --- optional propensity head (ablation) ---
        self.propensity_head = None
        if cfg.use_propensity_head:
            self.propensity_head = build_mlp(
                in_dim=trunk_out_dim,
                hidden_dims=(),
                out_dim=self.n_arms,
                activation=cfg.activation,
                dropout=0.0,
                norm="none",
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (batch, n_features)
        """
        z = self.trunk(x)

        if self.cfg.parameterization == "direct":
            assert self.heads is not None
            y_hat = torch.cat([h(z) for h in self.heads], dim=1)  # (batch, n_arms)
        else:
            assert self.y0_head is not None and self.delta_heads is not None
            y0 = self.y0_head(z)  # (batch, 1)
            if len(self.delta_heads) > 0:
                deltas = torch.cat([d(z) for d in self.delta_heads], dim=1)  # (batch, n_treatments)
                y_hat = torch.cat([y0, y0 + deltas], dim=1)
            else:
                y_hat = y0

        out: Dict[str, torch.Tensor] = {"y_hat": y_hat}
        if self.propensity_head is not None:
            out["t_logits"] = self.propensity_head(z)
        return out

    # ---------- inference helpers (numpy-friendly) ----------

    @torch.no_grad()
    def predict_potential_outcomes(
        self,
        x: Union[np.ndarray, torch.Tensor],
        *,
        device: Optional[torch.device] = None,
        batch_size: int = 8192,
    ) -> np.ndarray:
        """
        Returns potential outcome predictions for all arms.
        Shape: (n, n_arms)

        - For binary outcome: returns probabilities sigmoid(logits).
        - For continuous outcome: returns raw values.
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        x_t = torch.as_tensor(x, dtype=torch.float32)
        n = x_t.shape[0]
        outs: List[np.ndarray] = []

        for start in range(0, n, batch_size):
            xb = x_t[start : start + batch_size].to(device)
            y_hat = self.forward(xb)["y_hat"]
            if self.cfg.outcome_type == "binary":
                y_hat = torch.sigmoid(y_hat)
            outs.append(y_hat.detach().cpu().numpy())

        return np.concatenate(outs, axis=0)

    @torch.no_grad()
    def predict_uplift(
        self,
        x: Union[np.ndarray, torch.Tensor],
        *,
        device: Optional[torch.device] = None,
        batch_size: int = 8192,
    ) -> np.ndarray:
        """
        Returns tau_hat for each treatment arm vs control.
        Shape: (n, n_treatments)
        """
        po = self.predict_potential_outcomes(x, device=device, batch_size=batch_size)
        if self.cfg.n_treatments == 0:
            return np.zeros((po.shape[0], 0), dtype=np.float32)
        return po[:, 1:] - po[:, [0]]

    @torch.no_grad()
    def recommend_arm(
        self,
        x: Union[np.ndarray, torch.Tensor],
        *,
        device: Optional[torch.device] = None,
        batch_size: int = 8192,
        require_positive: bool = True,
    ) -> np.ndarray:
        """
        Simple greedy policy:
          - pick argmax tau_hat
          - if require_positive and max_uplift<=0 -> control (0)

        Returns arm indices in [0..n_arms-1].
        """
        tau = self.predict_uplift(x, device=device, batch_size=batch_size)
        if tau.shape[1] == 0:
            return np.zeros((tau.shape[0],), dtype=np.int64)

        best_treat = tau.argmax(axis=1) + 1
        if not require_positive:
            return best_treat.astype(np.int64)

        max_uplift = tau.max(axis=1)
        arms = best_treat.astype(np.int64)
        arms[max_uplift <= 0] = 0
        return arms
