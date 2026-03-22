import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# -----------------------------
# Scatter / segment ops (no extra deps)
# -----------------------------

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    src: (E, ...) ; index: (E,) in [0, dim_size-1]
    Returns: (dim_size, ...)
    """
    if index.ndim != 1:
        raise ValueError("index must be 1D")
    out_shape = (dim_size,) + src.shape[1:]
    out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


def scatter_max_1d(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    1D scatter max: src (E,), index (E,) -> out (dim_size,)
    Uses scatter_reduce_ if available, else falls back to sorting.
    """
    if src.ndim != 1 or index.ndim != 1:
        raise ValueError("scatter_max_1d expects 1D tensors")
    out = torch.full((dim_size,), -torch.inf, device=src.device, dtype=src.dtype)

    # PyTorch 1.12+/2.x: scatter_reduce_ exists
    if hasattr(out, "scatter_reduce_"):
        out.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
        return out

    # Fallback: sort by index and take max per group
    order = torch.argsort(index)
    idx_sorted = index[order]
    src_sorted = src[order]
    # compute max per segment manually
    unique_idx, start_pos = torch.unique_consecutive(idx_sorted, return_counts=False, return_inverse=False), None
    # (Fallback is rarely used; keep simple)
    for i in range(dim_size):
        mask = idx_sorted == i
        if mask.any():
            out[i] = src_sorted[mask].max()
    return out


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int, eps: float = 1e-12) -> torch.Tensor:
    """
    Softmax over segments defined by `index` (per dst-node softmax).
    scores: (E,)
    index:  (E,) segment id
    returns: (E,)
    """
    if scores.ndim != 1:
        raise ValueError("scores must be 1D")
    max_per = scatter_max_1d(scores, index, dim_size)              # (N,)
    exp = torch.exp(scores - max_per[index])                       # (E,)
    denom = scatter_sum(exp, index, dim_size).clamp_min(eps)        # (N,)
    return exp / denom[index]


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    edge_index: (2, E) with [src; dst]
    Returns edge_index with self-loops added (i->i).
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    device = edge_index.device
    self_loops = torch.arange(num_nodes, device=device, dtype=edge_index.dtype)
    self_loops = torch.stack([self_loops, self_loops], dim=0)  # (2, N)
    return torch.cat([edge_index, self_loops], dim=1)


# -----------------------------
# Breadth aggregator: attention over neighbors (Eq. 3–4)
# -----------------------------

class BreadthAttentionConv(nn.Module):
    """
    Implements the paper's breadth aggregator:

      e_ij = v^T tanh(Ws h_i + Wd h_j)              (Eq. 4)
      alpha_ij = softmax_j(e_ij) over neighbors j of i
      h_tilde_i = tanh( sum_j alpha_ij * (h_j W_msg) )            (Eq. 3)

    Edge convention: edge_index = [src; dst], aggregate incoming messages to dst.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        attn_dim: Optional[int] = None,
        *,
        add_loops: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim or out_dim
        self.add_loops = add_loops
        self.eps = eps

        # Message projection: W^(l)
        self.W_msg = nn.Linear(in_dim, out_dim, bias=False)

        # Attention projections: Ws, Wd and vector v
        self.Ws = nn.Linear(in_dim, self.attn_dim, bias=False)
        self.Wd = nn.Linear(in_dim, self.attn_dim, bias=False)
        self.v = nn.Parameter(torch.empty(self.attn_dim))
        nn.init.xavier_uniform_(self.Ws.weight)
        nn.init.xavier_uniform_(self.Wd.weight)
        nn.init.xavier_uniform_(self.W_msg.weight)
        nn.init.uniform_(self.v, -0.1, 0.1)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        h: (N, in_dim)
        edge_index: (2, E)
        returns h_tilde: (N, out_dim)
        """
        if h.ndim != 2 or h.shape[1] != self.in_dim:
            raise ValueError(f"h must be (N, {self.in_dim}), got {tuple(h.shape)}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")

        N = h.shape[0]
        if self.add_loops:
            edge_index = add_self_loops(edge_index, N)

        src, dst = edge_index[0].long(), edge_index[1].long()

        # Precompute projections for attention
        hs = self.Ws(h)              # (N, attn_dim)
        hd = self.Wd(h)              # (N, attn_dim)

        # e_ij for each edge (j=src, i=dst): v^T tanh(Ws h_i + Wd h_j)
        e = torch.tanh(hs[dst] + hd[src])            # (E, attn_dim)
        e = (e * self.v).sum(dim=-1)                 # (E,)

        alpha = segment_softmax(e, dst, dim_size=N, eps=self.eps)  # (E,)

        # messages: h_j W_msg
        msg = self.W_msg(h)[src]                     # (E, out_dim)
        msg = msg * alpha.unsqueeze(-1)              # weight per edge

        out = scatter_sum(msg, dst, dim_size=N)      # (N, out_dim)
        return torch.tanh(out)


class DepthMemoryAggregator(nn.Module):
    """
    Memory-based depth aggregator (LSTM-style) applied across GNN layers.

    Given h_tilde (from breadth conv), update per node:
      i = sigmoid(W_i h_tilde)
      f = sigmoid(W_f h_tilde)
      o = sigmoid(W_o h_tilde)
      C~ = tanh(W_c h_tilde)
      C = f ⊙ C_prev + i ⊙ C~
      h = o ⊙ tanh(C)

    (Matches the paper's depth aggregator equations; we implement vector gates.)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.W_i = nn.Linear(dim, dim, bias=True)
        self.W_f = nn.Linear(dim, dim, bias=True)
        self.W_o = nn.Linear(dim, dim, bias=True)
        self.W_c = nn.Linear(dim, dim, bias=True)

        for m in [self.W_i, self.W_f, self.W_o, self.W_c]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, h_tilde: torch.Tensor, c_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h_tilde: (N, dim)
        c_prev:  (N, dim)
        returns: (h, c)
        """
        if h_tilde.shape != c_prev.shape:
            raise ValueError("h_tilde and c_prev must have the same shape")

        i = torch.sigmoid(self.W_i(h_tilde))
        f = torch.sigmoid(self.W_f(h_tilde))
        o = torch.sigmoid(self.W_o(h_tilde))
        c_tilde = torch.tanh(self.W_c(h_tilde))

        c = f * c_prev + i * c_tilde
        h = o * torch.tanh(c)
        return h, c


@dataclass(frozen=True)
class GNUMEncoderConfig:
    n_features: int
    hidden_dim: int = 64
    num_layers: int = 2
    attn_dim: Optional[int] = None
    add_self_loops: bool = True
    dropout: float = 0.0


class GNUMEncoder(nn.Module):
    """
    Graph-based representation learning of GNUM:
      for each layer:
        h_tilde <- BreadthAttentionConv(h, edge_index)
        (h, c)  <- DepthMemoryAggregator(h_tilde, c)

    Input H^(0)=X, optionally projected to hidden_dim.
    """
    def __init__(self, cfg: GNUMEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Identity()
        if cfg.n_features != cfg.hidden_dim:
            self.input_proj = nn.Linear(cfg.n_features, cfg.hidden_dim)

        self.breadth_layers = nn.ModuleList([
            BreadthAttentionConv(
                in_dim=cfg.hidden_dim,
                out_dim=cfg.hidden_dim,
                attn_dim=cfg.attn_dim,
                add_loops=cfg.add_self_loops,
            )
            for _ in range(cfg.num_layers)
        ])

        self.depth_layers = nn.ModuleList([
            DepthMemoryAggregator(cfg.hidden_dim) for _ in range(cfg.num_layers)
        ])

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout and cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (N, n_features)
        edge_index: (2, E)
        returns: h_L (N, hidden_dim)
        """
        if x.ndim != 2 or x.shape[1] != self.cfg.n_features:
            raise ValueError(f"x must be (N, {self.cfg.n_features}), got {tuple(x.shape)}")

        h = self.input_proj(x)
        c = torch.zeros_like(h)

        for breadth, depth in zip(self.breadth_layers, self.depth_layers):
            h = self.dropout(h)
            h_tilde = breadth(h, edge_index)
            h, c = depth(h_tilde, c)

        return h
    

UpliftHeadType = Literal["ct", "pl"]


@dataclass(frozen=True)
class GNUMConfig:
    encoder: GNUMEncoderConfig
    head: UpliftHeadType = "ct"

    # CT: p = P(t=1) (в RCT константа), либо можно передавать propensity per-node
    p_treat: float = 0.5

    # PL: численная стабильность
    eps: float = 1e-8


class GNUMCTHead(nn.Module):
    """
    Class-transformed target head.
    Model outputs z_hat ~ E[z | x, G], and tau_hat is taken as z_hat.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_hat = self.lin(h).squeeze(-1)  # (N,)
        return {"z_hat": z_hat, "tau_hat": z_hat}


class GNUMPLHead(nn.Module):
    """
    Partial-label head:
      pA = P(S=[1,0,0] | x,G)   (Group A)
      pC = P(S=[0,0,1] | x,G)   (Group C)
      tau_hat = 1 - pA - pC
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lin_A = nn.Linear(hidden_dim, 1)
        self.lin_C = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        pA = torch.sigmoid(self.lin_A(h).squeeze(-1))
        pC = torch.sigmoid(self.lin_C(h).squeeze(-1))
        tau_hat = 1.0 - pA - pC
        return {"pA": pA, "pC": pC, "tau_hat": tau_hat}


class GNUM(nn.Module):
    """
    Full GNUM model = encoder + (CT or PL) head.
    """
    def __init__(self, cfg: GNUMConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = GNUMEncoder(cfg.encoder)

        if cfg.head == "ct":
            self.head = GNUMCTHead(cfg.encoder.hidden_dim)
        elif cfg.head == "pl":
            self.head = GNUMPLHead(cfg.encoder.hidden_dim)
        else:
            raise ValueError(f"Unknown head: {cfg.head}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x, edge_index)
        out = self.head(h)
        out["h"] = h
        return out

    @torch.no_grad()
    def predict_uplift(
        self,
        x: Union[np.ndarray, torch.Tensor],
        edge_index: Union[np.ndarray, torch.Tensor],
        *,
        node_idx: Optional[Union[np.ndarray, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Returns tau_hat for all nodes (or subset `node_idx`) as numpy.
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        ei = torch.as_tensor(edge_index, dtype=torch.long, device=device)

        out = self.forward(x_t, ei)
        tau = out["tau_hat"]

        if node_idx is not None:
            idx = torch.as_tensor(node_idx, dtype=torch.long, device=device)
            tau = tau[idx]

        return tau.detach().cpu().numpy()


# -----------------------------
# Losses
# -----------------------------

def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mask = mask.to(dtype=torch.float32)
    denom = mask.sum().clamp_min(eps)
    return (x * mask).sum() / denom


class GNUMCTLoss(nn.Module):
    """
    CT loss:
      z_i = y_obs * (t_i - p) / (p*(1-p))
      minimize MSE(z_hat, z) on labeled nodes
    """
    def __init__(self, p_treat: float = 0.5, eps: float = 1e-8):
        super().__init__()
        if not (0.0 < p_treat < 1.0):
            raise ValueError("p_treat must be in (0,1)")
        self.p = float(p_treat)
        self.eps = float(eps)

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        label_mask: Optional[torch.Tensor] = None,
        propensity: Optional[torch.Tensor] = None,  # optional p(x) вместо константы
    ) -> Dict[str, torch.Tensor]:
        z_hat = model_out["z_hat"]  # (N,)
        t = t.float()
        y = y.float()
        if propensity is None:
            p = torch.tensor(self.p, device=y.device, dtype=y.dtype)
        else:
            p = propensity.float().clamp(self.eps, 1.0 - self.eps)

        # z = y_obs * (t - p) / (p*(1-p))
        z = y * (t - p) / (p * (1.0 - p) + self.eps)

        mse = (z_hat - z) ** 2
        if label_mask is None:
            loss = mse.mean()
        else:
            loss = _masked_mean(mse, label_mask, eps=self.eps)

        return {"loss": loss, "mse": loss.detach()}


class GNUMPLLoss(nn.Module):
    """
    PL loss: two BCE losses with masks (ignore ambiguous samples per classifier).

    Mapping (Eq. 8):
      t=0,y=1 -> [1,0,0]
      t=0,y=0 -> [0,1,1]
      t=1,y=1 -> [1,1,0]
      t=1,y=0 -> [0,0,1]

    Classifier A:
      pos: [1,0,0] (t=0,y=1)
      neg: [0,1,1] or [0,0,1] (t=0,y=0 or t=1,y=0)
      ignore: [1,1,0] (t=1,y=1)

    Classifier C:
      pos: [0,0,1] (t=1,y=0)
      neg: [1,1,0] or [1,0,0] (t=1,y=1 or t=0,y=1)
      ignore: [0,1,1] (t=0,y=0)
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        label_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pA = model_out["pA"]
        pC = model_out["pC"]

        t = t.float()
        y = y.float()

        # base labeled mask (scarcity)
        base = torch.ones_like(y, dtype=torch.bool) if label_mask is None else label_mask.bool()

        # masks for combinations
        t0y1 = base & (t < 0.5) & (y > 0.5)
        t0y0 = base & (t < 0.5) & (y < 0.5)
        t1y1 = base & (t > 0.5) & (y > 0.5)
        t1y0 = base & (t > 0.5) & (y < 0.5)

        # --- classifier A ---
        maskA = t0y1 | t0y0 | t1y0
        yA = torch.zeros_like(y)
        yA[t0y1] = 1.0
        bceA = F.binary_cross_entropy(pA.clamp(self.eps, 1.0 - self.eps), yA, reduction="none")
        lossA = _masked_mean(bceA, maskA, eps=self.eps)

        # --- classifier C ---
        maskC = t1y0 | t1y1 | t0y1
        yC = torch.zeros_like(y)
        yC[t1y0] = 1.0
        bceC = F.binary_cross_entropy(pC.clamp(self.eps, 1.0 - self.eps), yC, reduction="none")
        lossC = _masked_mean(bceC, maskC, eps=self.eps)

        loss = lossA + lossC
        return {
            "loss": loss,
            "lossA": lossA.detach(),
            "lossC": lossC.detach(),
        }


# -----------------------------
# Data container + trainer (full-batch)
# -----------------------------

@dataclass
class GraphUpliftBatch:
    x: torch.Tensor              # (N, d)
    edge_index: torch.Tensor     # (2, E)
    t: torch.Tensor              # (N,) in {0,1}
    y: torch.Tensor              # (N,) observed outcome (binary for PL; any float for CT)
    label_mask: Optional[torch.Tensor] = None  # (N,) bool
    propensity: Optional[torch.Tensor] = None  # (N,) optional

    def to(self, device: torch.device) -> "GraphUpliftBatch":
        return GraphUpliftBatch(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            t=self.t.to(device),
            y=self.y.to(device),
            label_mask=None if self.label_mask is None else self.label_mask.to(device),
            propensity=None if self.propensity is None else self.propensity.to(device),
        )


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 5e-4  # paper uses L2 reg (λ=0.0005)
    max_epochs: int = 200
    grad_clip_norm: Optional[float] = 5.0
    device: Optional[str] = None
    verbose: bool = True


def train_gnum(
    model: GNUM,
    train_batch: GraphUpliftBatch,
    *,
    valid_batch: Optional[GraphUpliftBatch] = None,
    cfg: Optional[TrainConfig] = None,
) -> Dict[str, list]:
    if cfg is None:
        cfg = TrainConfig()

    device = torch.device(cfg.device) if cfg.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    train_batch = train_batch.to(device)
    if valid_batch is not None:
        valid_batch = valid_batch.to(device)

    if model.cfg.head == "ct":
        loss_fn = GNUMCTLoss(p_treat=model.cfg.p_treat, eps=model.cfg.eps)
    else:
        loss_fn = GNUMPLLoss(eps=model.cfg.eps)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train_loss": [], "valid_loss": []}

    def _run(batch: GraphUpliftBatch, training: bool) -> float:
        model.train(training)
        with torch.set_grad_enabled(training):
            out = model(batch.x, batch.edge_index)
            if model.cfg.head == "ct":
                losses = loss_fn(
                    out,
                    t=batch.t,
                    y=batch.y,
                    label_mask=batch.label_mask,
                    propensity=batch.propensity,
                )
            else:
                losses = loss_fn(out, t=batch.t, y=batch.y, label_mask=batch.label_mask)

            loss = losses["loss"]

            if training:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                opt.step()

        return float(loss.detach().cpu().item())

    for epoch in range(cfg.max_epochs):
        tr = _run(train_batch, training=True)
        history["train_loss"].append(tr)

        if valid_batch is not None:
            va = _run(valid_batch, training=False)
            history["valid_loss"].append(va)

        if cfg.verbose:
            msg = f"Epoch {epoch+1:03d}/{cfg.max_epochs} | train_loss={tr:.6f}"
            if valid_batch is not None:
                msg += f" | valid_loss={history['valid_loss'][-1]:.6f}"
            print(msg)

    return history
