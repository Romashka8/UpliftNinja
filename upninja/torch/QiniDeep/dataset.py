# ----------------------------------------------------------------------------------------------------------------------------------------

from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# ----------------------------------------------------------------------------------------------------------------------------------------


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
