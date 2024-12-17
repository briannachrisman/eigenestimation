# train.py
import torch
import einops
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from eigenestimation.eigenhora import EigenHora  # Assuming eigenestimation.py is in the same directory

eps = 1e-10

def Train(
    eigenmodel: EigenHora,
    jacobian_dataloader: DataLoader,
    lr: float,
    n_epochs: int,
    L0_penalty: float,
    device: str = 'cuda'
) -> None:
    # Collect parameters to optimize (excluding the model's own parameters)
    params_to_optimize = [*[t for name in eigenmodel.low_rank for t in eigenmodel.low_rank[name]]]
    optimizer: Optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    
    for epoch in range(n_epochs):
        sparsity_losses: float = 0.0
        reconstruction_losses: float = 0.0
        total_losses: float = 0.0
        n_batches: int = 0
        for jacobian in jacobian_dataloader:
            n_batches += 1
            jvp = eigenmodel(jacobian)
            reconstruction = eigenmodel.reconstruct(jvp.relu())
            jvp_einops_shape = ' '.join(["d" + str(i) for i in range(len(jvp.shape)-1)])
            L2_error = torch.stack([einops.einsum( # jvp_einops_shape = batch size
                (reconstruction[name] - jacobian[name])**2, f'{jvp_einops_shape} ... -> {jvp_einops_shape}') for name in jacobian], dim=0).sum(dim=0).mean()
            L0_error = einops.einsum((abs(jvp) + eps), '... f -> ...').mean()
            L = L2_error + L0_penalty * L0_error
            
            sparsity_losses += L0_error
            reconstruction_losses += L2_error
            total_losses += L
        
            L.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if epoch % max(1, round(n_epochs / 100)) == 0:
            avg_total_loss = total_losses / n_batches
            avg_sparsity_loss = sparsity_losses / n_batches
            avg_reconstruction_loss = reconstruction_losses / n_batches
            print(
                f'Epoch {epoch} : {avg_total_loss:.3f},  '
                f'Reconstruction Loss: {avg_reconstruction_loss:.3f},  '
                f'Sparsity Loss: {avg_sparsity_loss:.3f}'
            )
