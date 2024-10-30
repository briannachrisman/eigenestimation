import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Any
import einops
import gc

def TrainEigenEstimation(
    eigenmodel: nn.Module,
    dataloader: DataLoader,
    lr: float,
    n_epochs: int,
    lambda_penalty: float,
    device: str
) -> None:
    n_u_vectors: int = eigenmodel.n_u_vectors

    # Create a lower triangular mask for loss computation
    lower_triangular_mask: torch.Tensor = torch.tril(
        torch.ones(n_u_vectors, n_u_vectors, dtype=torch.bool), diagonal=-1
    )

    # Collect parameters to optimize (excluding the model's own parameters)
    params_to_optimize = [
        param for name, param in eigenmodel.named_parameters() if "model" not in name
    ]

    optimizer: Optimizer = torch.optim.SGD(params_to_optimize, lr=lr)

    for epoch in range(n_epochs):
        basis_losses: float = 0.0
        high_H_losses: float = 0.0
        total_losses: float = 0.0
        n_batches: int = 0

        for x in dataloader:
            # Normalize parameters at the start of each batch
            eigenmodel.normalize_parameters()

            n_batches += 1
            optimizer.zero_grad()  # Clear gradients

            # Forward pass
            dH_du, u_tensor = eigenmodel(x.to(device))
            dH_du = dH_du.flatten(0,-2)

            # Compute the loss components
            batch_size = dH_du.shape[0]
            mask = einops.repeat(
                lower_triangular_mask, 'k1 k2 -> batch k1 k2', batch=batch_size
            )

            # Compute cosine similarities between u vectors
            cosine_sims = u_tensor @ u_tensor.transpose(0, 1)  # Shape: [batch_size, k, k]

            # Compute magnitude products of dH_du
            mag_products = einops.einsum(
                dH_du, dH_du, 'batch k1, batch k2 -> batch k1 k2'
            )

            # Combine cosine similarities and magnitude products
            cosine_sims_mag = einops.einsum(
                cosine_sims, mag_products, 'k1 k2, batch k1 k2 -> batch k1 k2'
            )

            # Compute basis loss and high Hessian loss
            basis_loss = torch.abs(cosine_sims_mag[mask]).mean()
            high_H_loss = -(dH_du ** 2).mean()

            # Total loss
            L = high_H_loss + lambda_penalty * basis_loss

            # Accumulate losses
            basis_losses += basis_loss.item()
            high_H_losses += high_H_loss.item()
            total_losses += L.item()

            # Backpropagation and optimization step
            L.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Logging progress every 10% of total epochs
        if epoch % max(1, round(n_epochs / 10)) == 0:
            avg_total_loss = total_losses / n_batches
            avg_high_H_loss = high_H_losses / n_batches
            avg_basis_loss = basis_losses / n_batches
            print(
                f'Epoch {epoch} - Total Loss: {avg_total_loss:.3f}, '
                f'High Hessian Loss: {avg_high_H_loss:.3f},  '
                f'Basis Loss: {avg_basis_loss:.3f}'
            )
