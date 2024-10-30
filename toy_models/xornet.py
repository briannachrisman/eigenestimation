import einops
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Define the neural network for the XOR problem
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        # Input layer -> Hidden layer (2 neurons)
        self.fc1 = nn.Linear(2, 2, bias=True)  
        # Activation function
        self.activation = nn.GELU()  

    def forward(self, x):
        # Apply activation function after the first layer
        x = self.activation(self.fc1(x))  
        # Sum outputs and keep dimension for binary classification
        return x.sum(dim=1, keepdim=True)  

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
