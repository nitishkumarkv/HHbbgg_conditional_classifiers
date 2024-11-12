import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.models import GAT

class GAT_classifier(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_init_mlp_layers,
                 num_final_mlp_layers,
                 num_gat_layers,
                 num_message_passing_layers,
                 num_extra_feat,
                 output_dim,
                 global_pooling=global_add_pool,
                 dropout=0,
                 activation="leakyrelu",
                 v2_bool=True):
        super(GAT_classifier, self).__init__()

        self.global_pooling = global_pooling

        self.activation_dict = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'prelu': nn.PReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        
        # Initial MLP layers for the node features (num_init_mlp_layers)
        self.mlp_init_layers = nn.ModuleList()
        self.mlp_init_layers.append(nn.Linear(input_dim, hidden_dim))
        self.mlp_init_layers.append(self.activation_dict[activation])

        for _ in range(num_init_mlp_layers - 1):
            self.mlp_init_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp_init_layers.append(self.activation_dict[activation])


        # GAT layers with multi-head attention (num_gat_layers)
        self.gat_layers = nn.ModuleList()
        for _ in range(num_gat_layers):
            self.gat_layers.append(GAT(in_channels=hidden_dim, 
                                           hidden_channels=hidden_dim, # number of features (number of heads) of nodes after each layer
                                           num_layers=num_message_passing_layers,
                                           v2=v2_bool, 
                                           dropout=dropout,
                                           act=activation))


        # Final MLP layers for the nodes after the global pooling (num_final_mlp_layers)
        self.mlp_final_layers = nn.ModuleList()

        self.mlp_final_layers.append(nn.Linear(hidden_dim + num_extra_feat, hidden_dim))
        self.mlp_final_layers.append(self.activation_dict[activation])
        self.mlp_final_layers.append(nn.Dropout(dropout))

        # Additional final MLP layers
        for _ in range(num_final_mlp_layers - 2):
            self.mlp_final_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp_final_layers.append(self.activation_dict[activation])
            self.mlp_final_layers.append(nn.Dropout(dropout))
        
        self.mlp_final_layers.append(nn.Linear(hidden_dim, output_dim))
        

    def forward(self, data):
        x, edge_index, batch, extra_feat = data.x, data.edge_index, data.batch, data.extra_feat
        
        # Initial MLP to project node features to higher-dimensional space
        for mlp_layer in self.mlp_init_layers:
            x = mlp_layer(x)
        

        # GAT layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x=x, edge_index=edge_index)
        

        # Global pooling to aggregate node features into a single graph representation
        x = self.global_pooling(x, batch)
        

        # Final MLP layers for classification
        x = torch.cat([x, extra_feat], dim=1)

        for mlp_layer in self.mlp_final_layers:
            x = mlp_layer(x)
        
        return x


# test the GAT model
if __name__ == "__main__":

    data = torch.load("/net/scratch_cms3a/kasaraguppe/work/HHbbgg_classifier/HHbbgg_conditional_classifiers/data/gnn_data_lst/data_list.pt")
    data = data[0]

    input_dim = 11
    hidden_dim = 64
    num_init_mlp_layers = 2
    num_final_mlp_layers = 2
    num_gat_layers = 2
    num_message_passing_layers = 2
    num_extra_feat = 3
    output_dim = 4

    gat = GAT_classifier(input_dim = input_dim,
                         hidden_dim = hidden_dim,
                         num_init_mlp_layers = num_init_mlp_layers,
                         num_final_mlp_layers = num_final_mlp_layers,
                         num_gat_layers = num_gat_layers,
                         num_message_passing_layers = num_message_passing_layers,
                         num_extra_feat = num_extra_feat,
                         output_dim = output_dim
                         )

    out = gat(data)
    print(out)

    # print the model summary
    print("Model Summary: ")
    print(gat)
