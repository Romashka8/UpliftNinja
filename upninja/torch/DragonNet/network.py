from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

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
    activation: str = "elu",
    dropout: float = 0.0,
    norm: NormType = "none",
) -> nn.Sequential:
    """
    [Linear -> Norm -> Act -> Dropout] x len(hidden_dims) -> Linear(out_dim)
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
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class DragonNetConfig:
    n_features: int

    trunk_hidden_dims: Tuple[int, ...] = (200, 200, 200)  # у DragonNet часто широкие слои
    head_hidden_dims: Tuple[int, ...] = (100, 100)

    activation: str = "relu"
    dropout: float = 0.0
    norm: NormType = "none"

    outcome_type: OutcomeType = "binary"   # binary -> y-head outputs logits; continuous -> raw

    # Weights
    lambda_propensity: float = 1.0         # weight for propensity loss
    lambda_targeted: float = 1.0           # weight for targeted regularization

    # Targeted regularization switch
    use_targeted_regularization: bool = True

    # numerical stability
    eps: float = 1e-6


class DragonNet(nn.Module):
    """
    DragonNet:
      trunk: x -> h
      outcome heads: y0(h), y1(h)
      propensity head: t_logit(h)
      + trainable epsilon for targeted regularization (optional)

    Forward returns:
      y0, y1: (batch,) logits if binary else raw
      t_logit: (batch,) logit for P(T=1|X)
      p: (batch,) sigmoid(t_logit)
    """
    def __init__(self, cfg: DragonNetConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.n_features <= 0:
            raise ValueError("n_features must be positive")

        trunk_out = cfg.trunk_hidden_dims[-1] if len(cfg.trunk_hidden_dims) > 0 else cfg.n_features

        self.trunk = nn.Identity()
        if len(cfg.trunk_hidden_dims) > 0:
            self.trunk = build_mlp(
                in_dim=cfg.n_features,
                hidden_dims=cfg.trunk_hidden_dims[:-1],
                out_dim=trunk_out,
                activation=cfg.activation,
                dropout=cfg.dropout,
                norm=cfg.norm,
            )

        # outcome heads
        self.y0_head = build_mlp(
            in_dim=trunk_out,
            hidden_dims=cfg.head_hidden_dims,
            out_dim=1,
            activation=cfg.activation,
            dropout=cfg.dropout,
            norm=cfg.norm,
        )
        self.y1_head = build_mlp(
            in_dim=trunk_out,
            hidden_dims=cfg.head_hidden_dims,
            out_dim=1,
            activation=cfg.activation,
            dropout=cfg.dropout,
            norm=cfg.norm,
        )

        # propensity head (binary treatment)
        self.t_head = build_mlp(
            in_dim=trunk_out,
            hidden_dims=cfg.head_hidden_dims,
            out_dim=1,
            activation=cfg.activation,
            dropout=cfg.dropout,
            norm=cfg.norm,
        )

        # targeted regularization parameter epsilon (scalar)
        # In practice it’s often initialized at 0.
        self.epsilon = nn.Parameter(torch.zeros(1), requires_grad=cfg.use_targeted_regularization)

    def _check_x(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.cfg.n_features:
            raise ValueError(f"x must have shape (batch, {self.cfg.n_features}), got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._check_x(x)
        h = self.trunk(x)

        y0 = self.y0_head(h).squeeze(-1)
        y1 = self.y1_head(h).squeeze(-1)

        t_logit = self.t_head(h).squeeze(-1)
        p = torch.sigmoid(t_logit)

        return {"y0": y0, "y1": y1, "t_logit": t_logit, "p": p}

    # ---------- inference helpers ----------

    @torch.no_grad()
    def predict_potential_outcomes(
        self,
        x: Union[np.ndarray, torch.Tensor],
        *,
        device: Optional[torch.device] = None,
        batch_size: int = 8192,
    ) -> np.ndarray:
        """
        Returns array (n, 2): [Y0_hat, Y1_hat]
        For binary: probabilities; for continuous: raw.
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        x_t = torch.as_tensor(x, dtype=torch.float32)
        n = x_t.shape[0]
        outs: List[np.ndarray] = []

        for i in range(0, n, batch_size):
            xb = x_t[i : i + batch_size].to(device)
            out = self.forward(xb)
            y0, y1 = out["y0"], out["y1"]

            if self.cfg.outcome_type == "binary":
                y0 = torch.sigmoid(y0)
                y1 = torch.sigmoid(y1)

            po = torch.stack([y0, y1], dim=1)  # (batch,2)
            outs.append(po.detach().cpu().numpy())

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
        Returns tau_hat = Y1_hat - Y0_hat as (n,)
        """
        po = self.predict_potential_outcomes(x, device=device, batch_size=batch_size)
        return (po[:, 1] - po[:, 0]).astype(np.float32)

    @torch.no_grad()
    def predict_propensity(
        self,
        x: Union[np.ndarray, torch.Tensor],
        *,
        device: Optional[torch.device] = None,
        batch_size: int = 8192,
    ) -> np.ndarray:
        """
        Returns p_hat = P(T=1|X) as (n,)
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        x_t = torch.as_tensor(x, dtype=torch.float32)
        n = x_t.shape[0]
        outs: List[np.ndarray] = []

        for i in range(0, n, batch_size):
            xb = x_t[i : i + batch_size].to(device)
            p = self.forward(xb)["p"]
            outs.append(p.detach().cpu().numpy())

        return np.concatenate(outs, axis=0).astype(np.float32)