# ----------------------------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .network import DeepUpliftNetwork
from .loss import FactualLoss

# ----------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class TrainConfig:
    lr: float = 1e-2
    weight_decay: float = 1e-4
    max_epochs: int = 100
    grad_clip_norm: Optional[float] = None
    device: Optional[str] = None  # "cuda" / "cpu"
    verbose: bool = True


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
