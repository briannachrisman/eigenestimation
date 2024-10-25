import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import einops
from typing import Tuple

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(Autoencoder, self).__init__()

        # Define parameters for the autoencoder
        self.W: nn.Parameter = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.b: nn.Parameter = nn.Parameter(torch.zeros(input_dim))

        # Initialize W with Xavier normal
        nn.init.xavier_normal_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoding step
        h: torch.Tensor = einops.einsum(self.W, x, 'k f, b f -> b k')
        # Decoding step
        x_hat: torch.Tensor = einops.einsum(self.W, h, 'k f, b k -> b f')
        # Add bias and apply ReLU activation
        x_hat = x_hat + self.b
        return torch.relu(x_hat)

def GenerateTMSData(
    num_features: int, 
    num_datapoints: int, 
    sparsity: float, 
    batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, DataLoader]:
    
    # Initialize feature vectors with random values
    feature_vectors: np.ndarray = np.random.rand(num_datapoints, num_features)
    # Apply sparsity to the feature vectors
    feature_vectors = feature_vectors * (np.random.rand(num_datapoints, num_features) < sparsity)

    # Remove feature vectors that are all zeros
    non_zero_indices: np.ndarray = np.any(feature_vectors != 0, axis=1)
    feature_vectors = feature_vectors[non_zero_indices]

    # Convert feature vectors to numpy arrays
    data_inputs: np.ndarray = np.array(feature_vectors)
    data_labels: np.ndarray = np.array(feature_vectors)

    # Create tensors from numpy arrays and move to GPU
    X_tms: torch.Tensor = torch.tensor(data_inputs, dtype=torch.float32).to('cuda')
    Y_tms: torch.Tensor = torch.tensor(data_labels, dtype=torch.float32).to('cuda')

    # Create a DataLoader for the dataset
    dataloader_tms: DataLoader = DataLoader(
        TensorDataset(X_tms, Y_tms), 
        batch_size=batch_size, 
        shuffle=True, 
        generator=torch.Generator(device='cuda')
    )

    return X_tms, Y_tms, dataloader_tms
