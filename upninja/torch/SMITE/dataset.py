# ----------------------------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Literal, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


# ----------------------------------------------------------------------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------------------------------------------------------------------
