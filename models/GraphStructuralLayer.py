import torch.nn as nn
from torch_geometric.nn import TransformerConv


class GraphStructuralLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout, residual=True):
        super(GraphStructuralLayer, self).__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.residual = residual
        self.layer1 = TransformerConv(hidden_dim, hidden_dim // n_heads, heads=n_heads, concat=True, dropout=dropout)
        self.layer2 = TransformerConv(hidden_dim, hidden_dim // n_heads, heads=n_heads, concat=True, dropout=dropout)
        self.init_weights()

    def forward(self, x, edge_index):
        out1 = self.layer1(x, edge_index)
        out1 = self.activation(out1)
        out1 = self.layer2(out1, edge_index)
        out1 += x
        out1 = self.activation(out1)
        return out1

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()
