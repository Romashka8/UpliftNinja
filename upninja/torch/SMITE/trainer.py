# ----------------------------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .loss import SMITELoss
from .network import SMITE

# ----------------------------------------------------------------------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------------------------------------------------------------------


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

    device = (
        torch.device(train_cfg.device)
        if train_cfg.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)

    loss_fn = SMITELoss(model.cfg)
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(train_cfg.use_amp and device.type == "cuda")
    )

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
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), train_cfg.grad_clip_norm
                        )
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
            msg = f"Epoch {epoch + 1:03d}/{train_cfg.max_epochs} | train_loss={tr:.6f}"
            if valid_loader is not None:
                msg += f" | valid_loss={history['valid_loss'][-1]:.6f}"
            print(msg)

    return history


# ----------------------------------------------------------------------------------------------------------------------------------------
