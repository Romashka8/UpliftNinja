from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


OutcomeType = Literal["binary"]  # SMITE paper focuses on binary outcome
UpliftLossType = Literal["to", "ie"]  # Transformed Outcome / Indirect Estimation
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


def _as_float_tensor(x: Union[float, np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    if isinstance(x, (float, int)):
        return torch.tensor(float(x), dtype=torch.float32, device=device)
    t = torch.as_tensor(x, dtype=torch.float32, device=device)
    return t


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
    propensity_e: float = 0.5          # e(X)=P(T=1|X); in RCT often constant 0.5

    # numerical stability
    eps: float = 1e-6


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
            raise ValueError(f"x must have shape (batch, {self.cfg.n_features}), got {tuple(x.shape)}")

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

        x_rep = torch.cat([x, x], dim=0)                 # (2b, d)
        t_rep = torch.cat([t0, t1], dim=0)               # (2b, 1)
        xt = torch.cat([x_rep, t_rep], dim=1)            # (2b, d+1)

        logits = self.net(xt).squeeze(1)                 # (2b,)
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


class SMITELoss(nn.Module):
    """
    Implements SMITE objectives:

    outcome_loss: BCEWithLogits on factual head:
      logit_T = logit1 if T=1 else logit0

    uplift_loss:
      - "to": MSE( tau_hat , Z )
      - "ie": BCE( Π_Y(X) , T )

    total:
      (1-alpha)*uplift_loss + alpha*outcome_loss
    """
    def __init__(self, cfg: SMITEConfig):
        super().__init__()
        if not (0.0 <= cfg.alpha <= 1.0):
            raise ValueError("alpha must be in [0,1]")
        self.cfg = cfg
        self._bce_logits = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        e: Optional[Union[float, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        t: (batch,) in {0,1}
        y: (batch,) in {0,1}
        e: propensity, scalar or (batch,), defaults to cfg.propensity_e
        """
        logits0 = model_out["logits0"]
        logits1 = model_out["logits1"]
        mu0 = model_out["mu0"]
        mu1 = model_out["mu1"]
        tau_hat = model_out["tau_hat"]

        if t.ndim != 1 or y.ndim != 1:
            raise ValueError("t and y must have shape (batch,)")

        t = t.float()
        y = y.float()

        # factual selection (since T is 0/1, this equals picking correct head)
        factual_logits = torch.where(t > 0.5, logits1, logits0)
        outcome_loss = self._bce_logits(factual_logits, y)

        # uplift loss
        if self.cfg.uplift_loss == "to":
            e_use = self.cfg.propensity_e if e is None else e
            z = transformed_outcome(y=y, t=t, e=e_use, eps=self.cfg.eps)
            uplift_loss = F.mse_loss(tau_hat, z, reduction="mean")

        elif self.cfg.uplift_loss == "ie":
            # Π1 = μ1/(μ0+μ1)
            denom1 = (mu0 + mu1).clamp_min(self.cfg.eps)
            pi1 = (mu1 / denom1).clamp(self.cfg.eps, 1.0 - self.cfg.eps)

            # Π0 = (1-μ1)/((1-μ0)+(1-μ1))
            denom0 = ((1.0 - mu0) + (1.0 - mu1)).clamp_min(self.cfg.eps)
            pi0 = ((1.0 - mu1) / denom0).clamp(self.cfg.eps, 1.0 - self.cfg.eps)

            # Π_Y = Y*Π1 + (1-Y)*Π0
            pi_y = (y * pi1 + (1.0 - y) * pi0).clamp(self.cfg.eps, 1.0 - self.cfg.eps)

            uplift_loss = F.binary_cross_entropy(pi_y, t, reduction="mean")

        else:
            raise ValueError(f"Unknown uplift_loss: {self.cfg.uplift_loss}")

        total = (1.0 - self.cfg.alpha) * uplift_loss + self.cfg.alpha * outcome_loss

        return {
            "loss": total,
            "outcome_loss": outcome_loss.detach(),
            "uplift_loss": uplift_loss.detach(),
        }


class SMITEDataset(Dataset):
    """
    Minimal dataset for SMITE:
      X: float32 (n, d)
      T: int64/float (n,) in {0,1}
      Y: float32 (n,) in {0,1}

    Optionally:
      e: propensity (scalar or per-sample), (n,)
    """
    def __init__(
        self,
        x: Union[np.ndarray, torch.Tensor],
        t: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        *,
        e: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
    ):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.t = torch.as_tensor(t, dtype=torch.float32).view(-1)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1)

        if self.x.ndim != 2:
            raise ValueError("x must have shape (n, d)")
        if self.t.ndim != 1 or self.y.ndim != 1:
            raise ValueError("t and y must have shape (n,)")
        if len(self.x) != len(self.t) or len(self.x) != len(self.y):
            raise ValueError("x, t, y must have the same length")

        self.e = None
        if e is not None:
            if isinstance(e, (float, int)):
                self.e = float(e)
            else:
                e_t = torch.as_tensor(e, dtype=torch.float32).view(-1)
                if len(e_t) != len(self.x):
                    raise ValueError("per-sample e must have length n")
                self.e = e_t

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {"x": self.x[idx], "t": self.t[idx], "y": self.y[idx]}
        if isinstance(self.e, torch.Tensor):
            item["e"] = self.e[idx]
        return item


# -----------------------------
# Trainer
# -----------------------------

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    batch_size: int = 512
    grad_clip_norm: Optional[float] = 5.0
    use_amp: bool = False
    device: Optional[str] = None  # "cuda" / "cpu"
    verbose: bool = True


def train_smite(
    model: SMITE,
    train_loader: DataLoader,
    *,
    valid_loader: Optional[DataLoader] = None,
    train_cfg: Optional[TrainConfig] = None,
) -> Dict[str, List[float]]:
    """
    Training loop for SMITE.
    NOTE: Для uplift обычно early stopping делают по uplift-метрикам (Qini/AUUC),
    но здесь мы возвращаем лосс-историю — метрики легко добавить коллбеком в вашей библиотеке.
    """
    if train_cfg is None:
        train_cfg = TrainConfig()

    device = torch.device(train_cfg.device) if train_cfg.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    loss_fn = SMITELoss(model.cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(train_cfg.use_amp and device.type == "cuda"))

    history: Dict[str, List[float]] = {"train_loss": [], "valid_loss": []}

    def _run_epoch(loader: DataLoader, training: bool) -> float:
        if training:
            model.train()
        else:
            model.eval()

        total = 0.0
        n_batches = 0

        for batch in loader:
            x = batch["x"].to(device)
            t = batch["t"].to(device)
            y = batch["y"].to(device)
            e = batch.get("e", None)
            if e is not None:
                e = e.to(device)

            with torch.set_grad_enabled(training):
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    out = model(x)
                    losses = loss_fn(out, t=t, y=y, e=e)
                    loss = losses["loss"]

                if training:
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    if train_cfg.grad_clip_norm is not None:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
                    scaler.step(opt)
                    scaler.update()

            total += float(loss.detach().cpu().item())
            n_batches += 1

        return total / max(1, n_batches)

    for epoch in range(train_cfg.max_epochs):
        tr = _run_epoch(train_loader, training=True)
        history["train_loss"].append(tr)

        if valid_loader is not None:
            va = _run_epoch(valid_loader, training=False)
            history["valid_loss"].append(va)

        if train_cfg.verbose:
            msg = f"Epoch {epoch+1:03d}/{train_cfg.max_epochs} | train_loss={tr:.6f}"
            if valid_loader is not None:
                msg += f" | valid_loss={history['valid_loss'][-1]:.6f}"
            print(msg)

    return history
