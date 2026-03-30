# ----------------------------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

# ----------------------------------------------------------------------------------------------------------------------------------------

UpliftLossType = Literal["to", "ie"]  # Transformed Outcome / Indirect Estimation
NormType = Literal["none", "layernorm", "batchnorm"]

# ----------------------------------------------------------------------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------------------------------------------------------------------


def build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    *,
    activation: str = "relu",
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


# ----------------------------------------------------------------------------------------------------------------------------------------


def _as_float_tensor(
    x: Union[float, np.ndarray, torch.Tensor], device: torch.device
) -> torch.Tensor:
    if isinstance(x, (float, int)):
        return torch.tensor(float(x), dtype=torch.float32, device=device)
    t = torch.as_tensor(x, dtype=torch.float32, device=device)
    return t


# ----------------------------------------------------------------------------------------------------------------------------------------


def transformed_outcome(
    y: torch.Tensor,
    t: torch.Tensor,
    e: Union[float, torch.Tensor],
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Z = T*Y/e(X) - (1-T)*Y/(1-e(X))

    y: (batch,) float in {0,1}
    t: (batch,) float in {0,1}
    e: scalar or (batch,) propensity P(T=1|X)
    """
    if y.ndim != 1 or t.ndim != 1:
        raise ValueError("y and t must be 1D tensors of shape (batch,)")

    device = y.device
    e_t = _as_float_tensor(e, device)
    if e_t.ndim == 0:
        e_t = e_t.expand_as(y)
    elif e_t.ndim == 1 and e_t.shape[0] == y.shape[0]:
        pass
    else:
        raise ValueError("e must be a scalar or a 1D tensor of shape (batch,)")

    e_t = e_t.clamp(min=eps, max=1.0 - eps)
    t = t.float()
    y = y.float()

    return (t * y) / e_t - ((1.0 - t) * y) / (1.0 - e_t)


# ----------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SMITEConfig:
    """
    SMITE configuration (binary treatment, binary outcome).

    The SMITE paper feeds two inputs (X,1) and (X,0) into two identical sub-networks
    with shared weights, producing μ1(X) and μ0(X), then:
      uplift u(X) = μ1(X) - μ0(X).
    """

    n_features: int

    # MLP that maps concatenated (X, t_fixed) -> logit of P(Y=1|X,t)
    hidden_dims: Tuple[int, ...] = (64, 32)

    activation: str = "relu"
    dropout: float = 0.0
    norm: NormType = "none"

    # loss mixing coefficient: (1-alpha)*uplift_loss + alpha*outcome_loss
    alpha: float = 0.5

    uplift_loss: UpliftLossType = "to"  # "to" or "ie"
    propensity_e: float = 0.5  # e(X)=P(T=1|X); in RCT often constant 0.5

    # numerical stability
    eps: float = 1e-6


# ----------------------------------------------------------------------------------------------------------------------------------------


class SMITE(nn.Module):
    """
    SMITE (Siamese Model for Individual Treatment Effect).

    Internally there is ONE shared network f_theta([X, t]) -> logit.
    We compute:
      logit0 = f([X,0])
      logit1 = f([X,1])
      μ0 = sigmoid(logit0), μ1 = sigmoid(logit1)
      τ_hat = μ1 - μ0
    """

    def __init__(self, cfg: SMITEConfig):
        super().__init__()
        self.cfg = cfg

        # input dim: X plus 1 treatment indicator
        in_dim = cfg.n_features + 1
        self.net = build_mlp(
            in_dim=in_dim,
            hidden_dims=cfg.hidden_dims,
            out_dim=1,  # logit
            activation=cfg.activation,
            dropout=cfg.dropout,
            norm=cfg.norm,
        )

    def _check_x(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.cfg.n_features:
            raise ValueError(
                f"x must have shape (batch, {self.cfg.n_features}), got {tuple(x.shape)}"
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (batch, n_features)

        Returns:
          - logits0, logits1: (batch,)
          - mu0, mu1: (batch,) probabilities
          - tau_hat: (batch,) uplift = mu1 - mu0
        """
        self._check_x(x)
        device = x.device
        b = x.shape[0]

        # Single batched pass through shared net:
        # first b rows: t=0, next b rows: t=1
        t0 = torch.zeros((b, 1), dtype=x.dtype, device=device)
        t1 = torch.ones((b, 1), dtype=x.dtype, device=device)

        x_rep = torch.cat([x, x], dim=0)  # (2b, d)
        t_rep = torch.cat([t0, t1], dim=0)  # (2b, 1)
        xt = torch.cat([x_rep, t_rep], dim=1)  # (2b, d+1)

        logits = self.net(xt).squeeze(1)  # (2b,)
        logits0, logits1 = logits[:b], logits[b:]

        mu0 = torch.sigmoid(logits0)
        mu1 = torch.sigmoid(logits1)
        tau_hat = mu1 - mu0

        return {
            "logits0": logits0,
            "logits1": logits1,
            "mu0": mu0,
            "mu1": mu1,
            "tau_hat": tau_hat,
        }

    @torch.no_grad()
    def predict_potential_outcomes(
        self,
        x: Union[np.ndarray, torch.Tensor],
        *,
        device: Optional[torch.device] = None,
        batch_size: int = 8192,
    ) -> np.ndarray:
        """
        Returns [μ0(X), μ1(X)] as numpy array with shape (n, 2).
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        x_t = torch.as_tensor(x, dtype=torch.float32)
        n = x_t.shape[0]
        outs: List[np.ndarray] = []

        for start in range(0, n, batch_size):
            xb = x_t[start : start + batch_size].to(device)
            out = self.forward(xb)
            po = torch.stack([out["mu0"], out["mu1"]], dim=1)  # (batch,2)
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
        Returns τ_hat(X) = μ1(X) - μ0(X) as numpy array with shape (n,).
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        x_t = torch.as_tensor(x, dtype=torch.float32)
        n = x_t.shape[0]
        outs: List[np.ndarray] = []

        for start in range(0, n, batch_size):
            xb = x_t[start : start + batch_size].to(device)
            tau = self.forward(xb)["tau_hat"]
            outs.append(tau.detach().cpu().numpy())

        return np.concatenate(outs, axis=0)

    @torch.no_grad()
    def recommend_treatment(
        self,
        x: Union[np.ndarray, torch.Tensor],
        *,
        device: Optional[torch.device] = None,
        batch_size: int = 8192,
        require_positive: bool = True,
    ) -> np.ndarray:
        """
        Binary policy:
          - treat (1) if uplift>0
          - else control (0) if require_positive=True
          - else always treat (1)
        """
        tau = self.predict_uplift(x, device=device, batch_size=batch_size)
        if not require_positive:
            return np.ones_like(tau, dtype=np.int64)
        return (tau > 0).astype(np.int64)


# ----------------------------------------------------------------------------------------------------------------------------------------
