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
        self.W_in: nn.Parameter = nn.Parameter(torch.randn(input_dim, hidden_dim))
        #self.W_out: nn.Parameter = nn.Parameter(torch.randn(hidden_dim, input_dim))

        self.b: nn.Parameter = nn.Parameter(torch.zeros(input_dim))

        # Initialize W with Xavier normal
        nn.init.xavier_normal_(self.W_in)
        #nn.init.xavier_normal_(self.W_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoding step
        h: torch.Tensor = einops.einsum(self.W_in, x, 'f h, b f -> b h')
        # Decoding step
        x_hat: torch.Tensor = einops.einsum(self.W_in, h, 'f h, b h -> b f')
        # Add bias and apply ReLU activation
        x_hat = x_hat + self.b
        return torch.relu(x_hat)

def GenerateTMSData(
    num_features: int, 
    num_datapoints: int, 
    sparsity: float, 
    batch_size: int,
   # device: str
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
        #generator=torch.Generator(device=device)
    )

    return X_tms, Y_tms, dataloader_tms


def GenerateTMSDataParallel( num_features: int, 
    num_datapoints: int, 
    sparsity: float, 
    batch_size: int, n_networks: int):

    # Initialize feature vectors with random values
    feature_vectors: np.ndarray = np.random.rand(num_datapoints, num_features * n_networks)

    # Apply sparsity to the feature vectors
    feature_vectors = feature_vectors * (np.random.rand(num_datapoints, num_features * n_networks) < sparsity)
    print(feature_vectors.shape)
    # Remove feature vectors that are all zeros
    non_zero_indices: np.ndarray = np.any(feature_vectors != 0, axis=-1)
    print(non_zero_indices.shape)

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
        #generator=torch.Generator(device=device)
    )

    return X_tms, Y_tms, dataloader_tms



# Define the neural network for the XOR problem
class AutoencoderParallel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_networks) -> None:
        super(AutoencoderParallel, self).__init__()

        # Define parameters for the autoencoder
        self.W_in: nn.Parameter = nn.Parameter(torch.randn(n_networks*input_dim, n_networks*hidden_dim))
        #self.W_out: nn.Parameter = nn.Parameter(torch.randn(hidden_dim, input_dim))

        self.b: nn.Parameter = nn.Parameter(torch.zeros(n_networks*input_dim))

        # Initialize W with Xavier normal
        nn.init.xavier_normal_(self.W_in)
        #nn.init.xavier_normal_(self.W_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoding step
        h: torch.Tensor = einops.einsum(self.W_in, x, 'f h, b f -> b h')
        # Decoding step
        x_hat: torch.Tensor = einops.einsum(self.W_in, h, 'f h, b h -> b f')
        # Add bias and apply ReLU activation
        x_hat = x_hat + self.b
        return torch.relu(x_hat)




class MSELoss(nn.Module):
    def __init__(self, reduction: str = 'none') -> None:
        """
        KL Divergence Loss with a structure similar to MSELoss.
        
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default is 'mean'.
        """
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for KL Divergence Loss.
        
        Args:
            preds (torch.Tensor): Predicted logits or probabilities (not softmaxed).
            truth (torch.Tensor): Target probabilities.

        Returns:
            torch.Tensor: The computed KL Divergence Loss.
        """
        return nn.MSELoss(reduction='none')(truth, preds).mean(dim=-1)