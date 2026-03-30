# ----------------------------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Literal, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from .network import SMITEConfig, transformed_outcome

# ----------------------------------------------------------------------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------------------------------------------------------------------
