# ----------------------------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# ----------------------------------------------------------------------------------------------------------------------------------------


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

# ----------------------------------------------------------------------------------------------------------------------------------------
