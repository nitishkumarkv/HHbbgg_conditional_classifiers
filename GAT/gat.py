import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool

#Single GAT layer
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        #initialize weights
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, x_i, x_j):
        # func to calculate attention coefficients
        z2 = torch.cat([x_i, x_j], dim=-1)
        a = self.attn_fc(z2)
        return F.leaky_relu(a)

    def forward(self, x, edge_index):
        # Calculate z
        z = self.fc(x)
        
        row, col = edge_index
        alpha = self.edge_attention(z[row], z[col])

        # Apply attention coefficients and aggregate
        alpha = F.softmax(alpha, dim=1)
        out = torch.zeros_like(z)
        out.index_add_(0, row, alpha * z[col])

        return out

#Multiple Heads using single GAT layer
class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        #create empty list where all GAT layers can be appended
        self.heads = nn.ModuleList([GATLayer(in_dim, out_dim) for _ in range(num_heads)])
        #concat the outputs
        self.merge = merge

    def forward(self, x, edge_index):
        head_outs = [attn_head(x, edge_index) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=-1)
        else:
            return torch.mean(torch.stack(head_outs), dim=0)
    
# create GAT with two layers
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout_rate=0.085):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)

        # Global Pooling
        self.global_pool = global_mean_pool
        
        # MLP for classification
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 4),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, edge_index, batch):
        x = self.layer1(x, edge_index)
        x = F.elu(x)
        x = self.layer2(x, edge_index)

        # Apply global mean pooling
        x = self.global_pool(x, batch)
        
        # Apply MLP for classification
        x = self.mlp(x)

        return x