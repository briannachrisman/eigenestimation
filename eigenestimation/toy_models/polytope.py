import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import numpy as np 

class ReluNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_layers):
        super(ReluNetwork, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))

        # Hidden layers
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))
        # Initialize layers to xavier normal distributions
        
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        
        return self.layers[-1](x)#.log_softmax(dim=-1)
    
    


def GeneratePolytopeData(n_inputs, n_input_choices, n_samples):
    n_polytopes = n_input_choices**n_inputs
    polytope_choices = np.random.choice(list(range(n_polytopes)), n_polytopes, replace=False)
    keys = list(product(range(n_input_choices), repeat=n_inputs))
    lookup_dict = {k:p for k, p in zip(keys, polytope_choices)}
    X = n_input_choices*torch.rand((n_samples, n_inputs))
    

    # Lookup dict on each element of X
    y = torch.zeros(n_samples).long()#(n_samples, n_polytopes))
    for i, x in enumerate(X):
        y[i] = lookup_dict[tuple(x.floor().int().tolist())] #, lookup_dict[tuple(x.floor().int().tolist())]] = 1
    return X, y, lookup_dict



    


