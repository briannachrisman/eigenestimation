import einops
import torch
import torch.nn as nn
import itertools
from torch.utils.data import DataLoader, TensorDataset

# Define the neural network for the XOR problem
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        # Input layer -> Hidden layer (2 neurons)
        self.fc1 = nn.Linear(2, 2, bias=True)  
        self.fc2 = nn.Linear(2, 1, bias=True)  

        # Activation function
        self.activation = nn.GELU()  

    def forward(self, x):
        # Apply activation function after the first layer
        x = self.activation(self.fc1(x))  
        x = self.fc2(x)
        # Sum outputs and keep dimension for binary classification
        return x#.softmax(dim=-1)

def GenerateXORData(n_repeats, batch_size):
    # XOR dataset (inputs and expected outputs)
    X_xornet = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    Y_xornet = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    X_xornet_r = einops.repeat(X_xornet, 's f -> (s r) f', r=n_repeats)
    Y_xornet_r = einops.repeat(Y_xornet, 's f -> (s r) f', r=n_repeats)

    # Create a DataLoader for the XOR dataset
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader_xornet = DataLoader(
        TensorDataset(X_xornet_r, Y_xornet_r), 
        batch_size=batch_size, 
        shuffle=True,
        #generator=torch.Generator(device=device)
    )

    return X_xornet, Y_xornet, dataloader_xornet



def GenerateXORDataParallel(n_repeats, n_networks, batch_size):
    # XOR dataset (inputs and expected outputs)
    X_xornet = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    Y_xornet = torch.tensor([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=torch.float32)
    

    n_columns = X_xornet.size(0)  # Get the number of columns
    k = 2  # Length of combinations

    # Generate all ordered combinations with replacement of column indices
    column_indices = list(range(n_columns))
    combinations = list(itertools.product(column_indices, repeat=n_networks))  # Generate combinations with replacement

    # Use these combinations to index into the input tensor and create the desired output
    X_xornet_n_networks = torch.stack([X_xornet[list(combo),:] for combo in combinations]).flatten(-2,-1)



    Y_xornet_n_networks = X_xornet_n_networks[:,::2]!=X_xornet_n_networks[:,1::2]
   
    X_xornet_r = einops.repeat(X_xornet_n_networks, 's f -> (s r) f', r=n_repeats)
    Y_xornet_r = einops.repeat(Y_xornet_n_networks, 's f -> (s r) f', r=n_repeats).float()

    # Create a DataLoader for the XOR dataset
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader_xornet = DataLoader(
        TensorDataset(X_xornet_r, Y_xornet_r), 
        batch_size=batch_size, 
        shuffle=True,
        #generator=torch.Generator(device=device)
    )

    return X_xornet_n_networks, Y_xornet_n_networks, dataloader_xornet



# Define the neural network for the XOR problem
class XORNetParallel(nn.Module):
    def __init__(self, n_networks):
        super(XORNetParallel, self).__init__()
        # Input layer -> Hidden layer (2 neurons)
        self.fc1 = nn.Linear(2*n_networks, 2*n_networks, bias=True)  
        self.fc2 = nn.Linear(2*n_networks, n_networks, bias=True)  

        # Activation function
        self.activation = nn.GELU()  

    def forward(self, x):
        # Apply activation function after the first layer
        x = self.activation(self.fc1(x))  
        x = self.fc2(x)
        # Sum outputs and keep dimension for binary classification
        return x#.softmax(dim=-1)
