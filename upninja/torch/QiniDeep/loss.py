# ----------------------------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn

# ----------------------------------------------------------------------------------------------------------------------------------------


OutcomeType = Literal["binary", "continuous"]

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

# ----------------------------------------------------------------------------------------------------------------------------------------
