from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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


class FactualLoss(nn.Module):
    """
    Masked (factual) loss from QiniDeep/DUN:
    for each sample i, only prediction for assigned arm W_i contributes to loss.

    - binary: BCEWithLogits on selected logit
    - continuous: MSE on selected value
    """
    def __init__(self, outcome_type: OutcomeType = "binary"):
        super().__init__()
        self.outcome_type = outcome_type
        if outcome_type == "binary":
            self._loss = nn.BCEWithLogitsLoss(reduction="none")
        elif outcome_type == "continuous":
            self._loss = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown outcome_type: {outcome_type}")

    def forward(
        self,
        y_hat: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
        *,
        sample_weight: Optional[torch.Tensor] = None,
        propensity: Optional[torch.Tensor] = None,
        normalize_by_weight_sum: bool = True,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        y_hat: (batch, n_arms)
        w:     (batch,) int64 in [0..n_arms-1]
        y:     (batch,) float; for binary should be {0,1}

        sample_weight: optional weights (e.g., IPW).
        propensity: optional propensities P(W=w|X):
          - shape (batch, n_arms): we take propensity of assigned arm
          - shape (batch,): treated as already-selected propensity

        If propensity is provided and sample_weight is None, uses IPW:
            weight_i = 1 / propensity_i
        """
        if y_hat.ndim != 2:
            raise ValueError(f"Expected y_hat (batch, n_arms), got {tuple(y_hat.shape)}")
        w = w.long().view(-1)
        y = y.float().view(-1)

        # factual prediction
        y_factual = y_hat.gather(1, w.view(-1, 1)).squeeze(1)
        per_sample = self._loss(y_factual, y)

        if sample_weight is None and propensity is not None:
            if propensity.ndim == 2:
                p = propensity.gather(1, w.view(-1, 1)).squeeze(1)
            elif propensity.ndim == 1:
                p = propensity.view(-1)
            else:
                raise ValueError("propensity must have shape (batch,) or (batch, n_arms)")
            sample_weight = 1.0 / (p.clamp_min(eps))

        if sample_weight is not None:
            sw = sample_weight.float().view(-1)
            per_sample = per_sample * sw
            if normalize_by_weight_sum:
                return per_sample.sum() / (sw.sum().clamp_min(eps))
            return per_sample.mean()

        return per_sample.mean()


@dataclass
class TrainConfig:
    lr: float = 1e-2
    weight_decay: float = 1e-4
    max_epochs: int = 100
    grad_clip_norm: Optional[float] = None
    device: Optional[str] = None  # "cuda" / "cpu"
    verbose: bool = True


class UpliftDataset(Dataset):
    """
    Minimal dataset wrapper for uplift training.
    """
    def __init__(
        self,
        x: Union[np.ndarray, torch.Tensor],
        w: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        *,
        sample_weight: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.w = torch.as_tensor(w, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.sample_weight = None if sample_weight is None else torch.as_tensor(sample_weight, dtype=torch.float32)

        if self.x.ndim != 2:
            raise ValueError("x must have shape (n, n_features)")
        if self.w.ndim != 1 or self.y.ndim != 1:
            raise ValueError("w and y must have shape (n,)")
        if len(self.x) != len(self.w) or len(self.x) != len(self.y):
            raise ValueError("x, w, y must have the same length")
        if self.sample_weight is not None and len(self.sample_weight) != len(self.x):
            raise ValueError("sample_weight must have length n")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {"x": self.x[idx], "w": self.w[idx], "y": self.y[idx]}
        if self.sample_weight is not None:
            item["sample_weight"] = self.sample_weight[idx]
        return item


def train_deep_uplift_network(
    model: DeepUpliftNetwork,
    train_loader: DataLoader,
    *,
    valid_loader: Optional[DataLoader] = None,
    cfg: Optional[TrainConfig] = None,
) -> Dict[str, List[float]]:
    """
    Starter training loop:
      - optimizes masked factual loss (as described in QiniDeep)
      - optional propensity loss if enabled in model config

    IMPORTANT: for uplift you typically early-stop on an uplift metric (AUUC/Qini),
    but this loop only tracks factual losses; integrate your metric computation outside
    (e.g., via R's maq) and add early stopping if needed.
    """
    if cfg is None:
        cfg = TrainConfig()

    device = torch.device(cfg.device) if cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    factual_loss = FactualLoss(model.cfg.outcome_type)
    propensity_loss = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {"train_loss": [], "valid_loss": []}

    for epoch in range(cfg.max_epochs):
        model.train()
        total = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            w = batch["w"].to(device)
            y = batch["y"].to(device)
            sw = batch.get("sample_weight")
            if sw is not None:
                sw = sw.to(device)

            out = model(x)
            loss = factual_loss(out["y_hat"], w, y, sample_weight=sw)

            if model.propensity_head is not None and model.cfg.propensity_loss_weight > 0:
                loss = loss + model.cfg.propensity_loss_weight * propensity_loss(out["t_logits"], w)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            total += float(loss.detach().cpu().item())
            n_batches += 1

        train_l = total / max(1, n_batches)
        history["train_loss"].append(train_l)

        if valid_loader is not None:
            model.eval()
            v_total = 0.0
            v_batches = 0
            with torch.no_grad():
                for batch in valid_loader:
                    x = batch["x"].to(device)
                    w = batch["w"].to(device)
                    y = batch["y"].to(device)
                    sw = batch.get("sample_weight")
                    if sw is not None:
                        sw = sw.to(device)

                    out = model(x)
                    v_loss = factual_loss(out["y_hat"], w, y, sample_weight=sw)
                    if model.propensity_head is not None and model.cfg.propensity_loss_weight > 0:
                        v_loss = v_loss + model.cfg.propensity_loss_weight * propensity_loss(out["t_logits"], w)

                    v_total += float(v_loss.detach().cpu().item())
                    v_batches += 1
            history["valid_loss"].append(v_total / max(1, v_batches))

        if cfg.verbose:
            msg = f"Epoch {epoch+1:03d}/{cfg.max_epochs} | train_loss={train_l:.6f}"
            if valid_loader is not None:
                msg += f" | valid_loss={history['valid_loss'][-1]:.6f}"
            print(msg)

    return history
