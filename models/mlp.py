import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, num_layers, num_nodes, output_size, act_fn):
        super(MLP, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, num_nodes))
        layers.append(act_fn())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_nodes, num_nodes))
            layers.append(act_fn())

        # Output layer
        layers.append(nn.Linear(num_nodes, output_size))
        layers.append(nn.Softmax(dim=1))

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
