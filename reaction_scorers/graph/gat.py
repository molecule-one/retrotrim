"""Graph attention network for RGP."""

import torch
from torch import nn

from .gat_layer import MultiHeadGraphConvLayer


class GAT(nn.Module):
    """Embed atoms, attend over reaction bonds, max-pool, then predict a logit.

    Residual connections occur on every second attention layer.
    """

    def __init__(
        self,
        atom_features,
        bond_features,
        hidden_dim=512,
        n_conv_layers=4,
        n_fc_layers=3,
        residual=True,
        fc_hidden_dim=512,
        batch_norm=True,
        embedding_bias=True,
        output_dim=None,
        conv_layer_cls=MultiHeadGraphConvLayer,
    ):
        super().__init__()
        self.atom_embedding = nn.Linear(atom_features, hidden_dim, bias=embedding_bias)
        self.conv_layers = []
        for i in range(n_conv_layers):
            layer = conv_layer_cls(
                bond_dim=bond_features,
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                residual=residual and i % 2 != 0,
            )
            self.conv_layers.append(layer)
            self.add_module(f"MultiHeadGraphConv_{i + 1}", layer)
        self.fc_layers = []
        for i in range(n_fc_layers):
            input_dim = hidden_dim if i == 0 else fc_hidden_dim
            final = i == n_fc_layers - 1
            layer = nn.Linear(input_dim, (output_dim or 1) if final else fc_hidden_dim)
            if batch_norm and not final:
                layer = nn.Sequential(layer, nn.BatchNorm1d(fc_hidden_dim))
            self.fc_layers.append(layer)
            self.add_module(f"FullyConnected_{i + 1}", layer)

    def forward(self, batch):
        atoms, bonds = batch["atom"], batch["bond"]
        x = self.atom_embedding(atoms)
        adjacency = torch.max(bonds, dim=-1)[0]
        soft_mask = (1.0 - adjacency) * -1e9
        atom_mask = torch.max(adjacency, dim=-1)[0].unsqueeze(-1)
        for layer in self.conv_layers:
            x = layer(x, bonds, adjacency, soft_mask, atom_mask.expand(x.shape))
        x = torch.where(
            atom_mask.expand(x.shape).bool(), x, x.new_tensor(-float("inf"))
        )
        x = x.max(dim=1)[0]
        for layer in self.fc_layers[:-1]:
            x = torch.relu(layer(x))
        return self.fc_layers[-1](x)[:, 0]
