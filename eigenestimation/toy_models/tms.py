import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import einops
from typing import Tuple
import itertools

class AutoencoderSymmetric(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(AutoencoderSymmetric, self).__init__()

        # Define parameters for the autoencoder
        self.W_in: nn.Parameter = nn.Parameter(torch.randn(input_dim, hidden_dim))
        #self.W_out = self.W_in.transpose(0,1)

        self.b: nn.Parameter = nn.Parameter(torch.zeros(input_dim))

        # Initialize W with Xavier normal
        nn.init.xavier_normal_(self.W_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoding step
        h: torch.Tensor = einops.einsum(self.W_in, x, 'f h, b f -> b h')
        # Decoding step
        x_hat: torch.Tensor = einops.einsum(self.W_in.transpose(0,1), h, 'h f, b h -> b f')
        # Add bias and apply ReLU activation
        x_hat = x_hat + self.b
        return torch.relu(x_hat)

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(Autoencoder, self).__init__()

        # Define parameters for the autoencoder
        self.W_in: nn.Parameter = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W_out: nn.Parameter = nn.Parameter(torch.randn(hidden_dim, input_dim))

        self.b: nn.Parameter = nn.Parameter(torch.zeros(input_dim))

        # Initialize W with Xavier normal
        nn.init.xavier_normal_(self.W_in)
        nn.init.xavier_normal_(self.W_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoding step
        h: torch.Tensor = einops.einsum(self.W_in, x, 'f h, b f -> b h')
        # Decoding step
        x_hat: torch.Tensor = einops.einsum(self.W_out, h, 'h f, b h -> b f')
        # Add bias and apply ReLU activation
        x_hat = x_hat + self.b
        return torch.relu(x_hat)
    
    
class SingleHiddenLayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim:int) -> None:
        super(SingleHiddenLayerPerceptron, self).__init__()

        # Define parameters for the autoencoder
        self.W_in: nn.Parameter = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W_out: nn.Parameter = nn.Parameter(torch.randn(hidden_dim, output_dim))

        self.b: nn.Parameter = nn.Parameter(torch.zeros(output_dim))

        # Initialize W with Xavier normal
        #nn.init.xavier_normal_(self.W_in)
        #nn.init.xavier_normal_(self.W_out)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoding step
        h: torch.Tensor = einops.einsum(self.W_in, x, 'f h, b f -> b h')
        # Decoding step
        x_hat: torch.Tensor = einops.einsum(self.W_out, h, 'h o, b h -> b o')
        # Add bias and apply ReLU activation
        x_hat = x_hat + self.b
        return torch.relu(x_hat)



# Define the neural network for the XOR problem
class AutoencoderParallel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_networks) -> None:
        super(AutoencoderParallel, self).__init__()

        # Define parameters for the autoencoder
        self.W_in: nn.Parameter = nn.Parameter(torch.randn(n_networks*input_dim, n_networks*hidden_dim))
        self.W_out: nn.Parameter = nn.Parameter(torch.randn(n_networks*hidden_dim, n_networks*input_dim))

        self.b: nn.Parameter = nn.Parameter(torch.zeros(n_networks*input_dim))

        # Initialize W with Xavier normal
        nn.init.xavier_normal_(self.W_in)
        #nn.init.xavier_normal_(self.W_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoding step
        h: torch.Tensor = einops.einsum(self.W_in, x, 'f h, b f -> b h')
        # Decoding step
        x_hat: torch.Tensor = einops.einsum(self.W_out, h, 'h f, b h -> b f')
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