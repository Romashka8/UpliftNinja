from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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
    

class DragonNetLoss(nn.Module):
    """
    Total loss:
      L = L_outcome + lambda_propensity * L_prop + lambda_targeted * L_targeted (optional)

    Supports optional sample_weight (e.g., IPW).
    """
    def __init__(self, cfg: DragonNetConfig):
        super().__init__()
        self.cfg = cfg
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")

        if cfg.lambda_propensity < 0 or cfg.lambda_targeted < 0:
            raise ValueError("lambda weights must be non-negative")

    def forward(
        self,
        model: DragonNet,
        model_out: Dict[str, torch.Tensor],
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        sample_weight: Optional[torch.Tensor] = None,
        normalize_by_weight_sum: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        t: (batch,) in {0,1}
        y: (batch,) float; for binary should be {0,1}
        sample_weight: optional (batch,)
        """
        y0 = model_out["y0"]
        y1 = model_out["y1"]
        t_logit = model_out["t_logit"]
        p = model_out["p"]

        if t.ndim != 1 or y.ndim != 1:
            raise ValueError("t and y must be 1D (batch,)")

        t = t.float()
        y = y.float()

        # -------- factual outcome prediction --------
        y_f_logit = torch.where(t > 0.5, y1, y0)  # (batch,)

        if self.cfg.outcome_type == "binary":
            per_outcome = self.bce_logits(y_f_logit, y)  # (batch,)
        else:
            per_outcome = self.mse(y_f_logit, y)

        # -------- propensity loss --------
        per_prop = self.bce_logits(t_logit, t)

        # -------- targeted regularization --------
        # use probabilities for binary; raw for continuous
        if self.cfg.outcome_type == "binary":
            y0_prob = torch.sigmoid(y0)
            y1_prob = torch.sigmoid(y1)
            y_f = torch.where(t > 0.5, y1_prob, y0_prob)
        else:
            y_f = torch.where(t > 0.5, y1, y0)

        eps = self.cfg.eps
        p_clip = p.clamp(eps, 1.0 - eps)
        clever_h = (t / p_clip) - ((1.0 - t) / (1.0 - p_clip))  # (batch,)

        y_tilde = y_f + model.epsilon.view(1) * clever_h

        # targeted term uses MSE as a stabilizer (common implementation choice)
        per_targeted = self.mse(y_tilde, y)

        # -------- combine + weighting --------
        per_total = per_outcome + self.cfg.lambda_propensity * per_prop
        if self.cfg.use_targeted_regularization:
            per_total = per_total + self.cfg.lambda_targeted * per_targeted
        else:
            per_targeted = per_targeted.detach()  # for logging consistency

        if sample_weight is not None:
            sw = sample_weight.float().view(-1)
            if sw.shape[0] != per_total.shape[0]:
                raise ValueError("sample_weight must have shape (batch,)")

            per_total = per_total * sw
            per_outcome = per_outcome * sw
            per_prop = per_prop * sw
            per_targeted = per_targeted * sw

            if normalize_by_weight_sum:
                denom = sw.sum().clamp_min(eps)
                loss = per_total.sum() / denom
                out_l = per_outcome.sum() / denom
                prop_l = per_prop.sum() / denom
                tar_l = per_targeted.sum() / denom
            else:
                loss = per_total.mean()
                out_l = per_outcome.mean()
                prop_l = per_prop.mean()
                tar_l = per_targeted.mean()
        else:
            loss = per_total.mean()
            out_l = per_outcome.mean()
            prop_l = per_prop.mean()
            tar_l = per_targeted.mean()

        return {
            "loss": loss,
            "outcome_loss": out_l.detach(),
            "propensity_loss": prop_l.detach(),
            "targeted_loss": tar_l.detach(),
            "epsilon": model.epsilon.detach().clone(),
        }
    

from torch.utils.data import Dataset, DataLoader

class DragonNetDataset(Dataset):
    """
    Minimal dataset for DragonNet:
      X: (n,d)
      T: (n,) in {0,1}
      Y: (n,)
    Optional: sample_weight (n,)
    """
    def __init__(
        self,
        x: Union[np.ndarray, torch.Tensor],
        t: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        *,
        sample_weight: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.t = torch.as_tensor(t, dtype=torch.float32).view(-1)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1)
        self.sample_weight = None if sample_weight is None else torch.as_tensor(sample_weight, dtype=torch.float32).view(-1)

        if self.x.ndim != 2:
            raise ValueError("x must have shape (n, d)")
        if len(self.t) != len(self.x) or len(self.y) != len(self.x):
            raise ValueError("x, t, y must have the same length")
        if self.sample_weight is not None and len(self.sample_weight) != len(self.x):
            raise ValueError("sample_weight must have length n")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch = {"x": self.x[idx], "t": self.t[idx], "y": self.y[idx]}
        if self.sample_weight is not None:
            batch["sample_weight"] = self.sample_weight[idx]
        return batch


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    grad_clip_norm: Optional[float] = 5.0
    use_amp: bool = False
    device: Optional[str] = None
    verbose: bool = True


def train_dragonnet(
    model: DragonNet,
    train_loader: DataLoader,
    *,
    valid_loader: Optional[DataLoader] = None,
    cfg: Optional[TrainConfig] = None,
) -> Dict[str, List[float]]:
    if cfg is None:
        cfg = TrainConfig()

    device = torch.device(cfg.device) if cfg.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    loss_fn = DragonNetLoss(model.cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    history: Dict[str, List[float]] = {"train_loss": [], "valid_loss": []}

    def _run_epoch(loader: DataLoader, training: bool) -> float:
        model.train(training)
        total = 0.0
        n_batches = 0

        for batch in loader:
            x = batch["x"].to(device)
            t = batch["t"].to(device)
            y = batch["y"].to(device)
            sw = batch.get("sample_weight")
            if sw is not None:
                sw = sw.to(device)

            with torch.set_grad_enabled(training):
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    out = model(x)
                    losses = loss_fn(model, out, t=t, y=y, sample_weight=sw)
                    loss = losses["loss"]

                if training:
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    if cfg.grad_clip_norm is not None:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    scaler.step(opt)
                    scaler.update()

            total += float(loss.detach().cpu().item())
            n_batches += 1

        return total / max(1, n_batches)

    for epoch in range(cfg.max_epochs):
        tr = _run_epoch(train_loader, training=True)
        history["train_loss"].append(tr)

        if valid_loader is not None:
            va = _run_epoch(valid_loader, training=False)
            history["valid_loss"].append(va)

        if cfg.verbose:
            msg = f"Epoch {epoch+1:03d}/{cfg.max_epochs} | train_loss={tr:.6f}"
            if valid_loader is not None:
                msg += f" | valid_loss={history['valid_loss'][-1]:.6f}"
            print(msg)

    return history
