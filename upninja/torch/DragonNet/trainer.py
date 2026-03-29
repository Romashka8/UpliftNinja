# ----------------------------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import torch
from torch.utils.data import DataLoader

from .network import DragonNet
from .loss import DragonNetLoss

OutcomeType = Literal["binary", "continuous"]
NormType = Literal["none", "layernorm", "batchnorm"]

# ----------------------------------------------------------------------------------------------------------------------------------------


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    grad_clip_norm: Optional[float] = 5.0
    use_amp: bool = False
    device: Optional[str] = None
    verbose: bool = True


# ----------------------------------------------------------------------------------------------------------------------------------------


def train_dragonnet(
    model: DragonNet,
    train_loader: DataLoader,
    *,
    valid_loader: Optional[DataLoader] = None,
    cfg: Optional[TrainConfig] = None,
) -> Dict[str, List[float]]:
    if cfg is None:
        cfg = TrainConfig()

    device = (
        torch.device(cfg.device)
        if cfg.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)

    loss_fn = DragonNetLoss(model.cfg)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
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
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.grad_clip_norm
                        )
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
            msg = f"Epoch {epoch + 1:03d}/{cfg.max_epochs} | train_loss={tr:.6f}"
            if valid_loader is not None:
                msg += f" | valid_loss={history['valid_loss'][-1]:.6f}"
            print(msg)

    return history


# ----------------------------------------------------------------------------------------------------------------------------------
