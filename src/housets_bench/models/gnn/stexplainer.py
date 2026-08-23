"""STExplainer — Explainable Spatio-Temporal Graph Neural Networks (HKUDS, CIKM 2023).

Reference: https://github.com/HKUDS/STExplainer  (arXiv:2310.17149)

Run here as a **forecaster only**: the paper's headline contribution is its
learned graph-structure explanation (evaluated on sparsity/fidelity metrics
this benchmark has no scoring path for), so that explanation output is not
surfaced anywhere outside this module — only the forecast is. The Graph
Information Bottleneck (GIB) structure-distillation term is kept as an
internal training-loss regularizer on a learned per-edge importance mask
(standard GNNExplainer-style size + entropy penalties: encourage the mask to
be small and confident, i.e. compress the graph to task-relevant edges),
added via ``_graph_forward_train``. The network itself is a standard STG
attention encoder (temporal self-attention + graph propagation over the
masked adjacency, reusing :class:`~housets_bench.models.gnn.stgformer.GraphPropagate`)
with a small positional-fusion decoder (a learnable per-horizon positional
embedding fused into the pooled encoding before the output head).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from housets_bench.models.gnn.gnn_forecaster import GNNForecasterBase
from housets_bench.models.gnn.stgformer import GraphPropagate
from housets_bench.models.registry import register


class EdgeMask(nn.Module):
    """Learned per-edge importance mask (GIB structure distillation)."""

    def __init__(self, edge_index: torch.Tensor, n_nodes: int) -> None:
        super().__init__()
        self.register_buffer("src", edge_index[0].long())
        self.register_buffer("dst", edge_index[1].long())
        self.n_nodes = int(n_nodes)
        self.logit = nn.Parameter(torch.zeros(edge_index.shape[1]))

    def apply(self, A_norm: torch.Tensor):
        mask = torch.sigmoid(self.logit)  # [E]
        A_masked = torch.zeros_like(A_norm)
        if self.src.numel() > 0:
            vals = A_norm[self.src, self.dst] * mask
            A_masked[self.src, self.dst] = vals
        return A_masked, mask


class STAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, order: int, dropout: float) -> None:
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.prop = GraphPropagate(d_model, order, dropout)
        self.prop_norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, A_masked: torch.Tensor) -> torch.Tensor:
        B, L, N, D = h.shape
        h_bn = h.permute(0, 2, 1, 3).reshape(B * N, L, D)
        attn_out, _ = self.temporal_attn(h_bn, h_bn, h_bn)
        h_bn = self.temporal_norm(h_bn + attn_out)
        h_t = h_bn.reshape(B, N, L, D).permute(0, 2, 1, 3)

        h_sp = self.prop(h_t, A_masked)
        return self.prop_norm(h_t + h_sp)


class STExplainerNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        pred_len: int,
        n_nodes: int,
        seq_len: int,
        edge_index: torch.Tensor,
        *,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        order: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)

        self.in_proj = nn.Linear(int(input_dim), int(d_model))
        self.edge_mask = EdgeMask(edge_index, n_nodes)
        self.blocks = nn.ModuleList([
            STAttnBlock(int(d_model), int(n_heads), int(order), float(dropout))
            for _ in range(int(n_layers))
        ])
        self.enc_pool = nn.Linear(int(d_model) * int(seq_len), int(d_model))
        self.pos_emb = nn.Parameter(torch.randn(self.pred_len, int(d_model)) * 0.1)
        self.step_head = nn.Linear(int(d_model), self.out_dim)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor):
        B, L, N, _ = x.shape
        h = self.in_proj(x)
        A_masked, mask = self.edge_mask.apply(A_norm)

        for block in self.blocks:
            h = block(h, A_masked)

        h_flat = h.permute(0, 2, 1, 3).reshape(B, N, -1)
        enc = self.enc_pool(h_flat)  # [B, N, d_model]

        outs = []
        for t in range(self.pred_len):
            outs.append(self.step_head(enc + self.pos_emb[t]))  # [B, N, out_dim]
        out = torch.stack(outs, dim=1)  # [B, pred_len, N, out_dim]
        return out, mask


@register("stexplainer")
class STExplainerForecaster(GNNForecasterBase):
    """STExplainer forecaster (forecast-only — GIB mask kept internal, no explanation output)."""

    name: str = "stexplainer"
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    order: int = 2
    dropout: float = 0.1
    gib_size_weight: float = 1e-3
    gib_entropy_weight: float = 1e-3

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        edge_index = self._A_raw.coalesce().indices()
        return STExplainerNet(
            input_dim=len(bundle.x_cols),
            out_dim=len(bundle.y_cols),
            pred_len=int(bundle.raw.spec.pred_len),
            n_nodes=n_nodes,
            seq_len=int(bundle.raw.spec.seq_len),
            edge_index=edge_index,
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            n_layers=int(self.n_layers),
            order=int(self.order),
            dropout=float(self.dropout),
        )

    def _graph_forward(self, net, x):
        out, _mask = net(x, self._A_norm.to(x.device).to_dense())
        return out

    def _graph_forward_train(self, net, x, y_true, progress):
        out, mask = net(x, self._A_norm.to(x.device).to_dense())
        if mask.numel() == 0:
            return out, out.new_zeros(())
        mask_c = mask.clamp(1e-6, 1.0 - 1e-6)
        size_loss = mask_c.mean()
        entropy_loss = (-mask_c * torch.log(mask_c) - (1 - mask_c) * torch.log(1 - mask_c)).mean()
        aux_loss = float(self.gib_size_weight) * size_loss + float(self.gib_entropy_weight) * entropy_loss
        return out, aux_loss
