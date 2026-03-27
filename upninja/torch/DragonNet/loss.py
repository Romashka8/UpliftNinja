from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from .network import DragonNetConfig, DragonNet

OutcomeType = Literal["binary", "continuous"]
NormType = Literal["none", "layernorm", "batchnorm"]


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