"""Bond-aware graph attention."""
from __future__ import annotations

import torch
from torch import nn


class MultiHeadGraphConvLayer(nn.Module):
    def __init__(
        self,
        bond_dim: int,
        input_dim: int,
        output_dim: int,
        residual: bool,
        att_heads: int = 8,
        att_dim: int = 64,
        node_norm: bool = True,
    ):
        """Bond-aware multi-head attention with optional per-node normalization."""
        super(MultiHeadGraphConvLayer, self).__init__()
        self.n_att = att_heads
        self.att_dim = att_dim
        self.node_norm = node_norm
        self.atoms_att = nn.Linear(input_dim, att_dim)
        self.final_att = nn.Linear(att_dim * 2 + bond_dim, att_heads)
        self.conv_layers = []
        if output_dim % att_heads != 0:
            raise ValueError(
                f"Output dimension {output_dim} must be divisible by {att_heads} heads."
            )
        for i in range(att_heads):
            conv = nn.Linear(input_dim, int(output_dim / att_heads))
            self.conv_layers.append(conv)
            setattr(self, f"graph_conv_{i + 1}", conv)
        self.residual = residual

    def forward(
        self, x, adj, mask, soft_mask, node_mask=None, apply_activation: bool = True
    ):
        """Attend over real edges and mask padding before node normalization."""
        x_att = torch.relu(self.atoms_att(x))
        x_att_shape = adj.shape[:-1] + (x_att.shape[-1],)
        x_rows = torch.unsqueeze(x_att, 1).expand(x_att_shape)
        x_cols = torch.unsqueeze(x_att, 2).expand(x_att_shape)
        x_att = torch.cat([x_rows, x_cols, adj], dim=-1)
        x_att = self.final_att(x_att)
        head_outs = []
        for i, conv in enumerate(self.conv_layers):
            att = x_att[:, :, :, i]
            att = torch.softmax(att + soft_mask, dim=-1) * mask
            out = torch.bmm(att, x)
            out = conv(out)
            head_outs.append(out)
        out = torch.cat(head_outs, dim=-1)
        if self.node_norm:
            if node_mask is None:
                raise ValueError("When using node_norm, node_mask must be provided")
            mean_val = torch.mean(out, dim=-1).unsqueeze(-1) * node_mask
            std_val = torch.std(out, dim=-1).unsqueeze(-1) * node_mask + (1 - node_mask)
            out = (out - mean_val) / std_val
        if self.residual:
            out = x + out
        if apply_activation:
            out = torch.relu(out)
        return out
