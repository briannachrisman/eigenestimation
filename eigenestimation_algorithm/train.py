import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Any
import einops
import gc

def TrainEigenEstimation(
    eigenmodel: nn.Module,
    x_dataloader: DataLoader,
    lr: float,
    n_epochs: int,
    lambda_penalty: float,
    u_batch_size = 20,
    device: str = 'cuda'
) -> None:

  
    # Create a lower triangular mask for loss computation
    lower_triangular_mask: torch.Tensor = torch.tril(
        torch.ones(eigenmodel.n_u_vectors, eigenmodel.n_u_vectors, dtype=torch.bool), diagonal=-1
    ).to(device)

    # Collect parameters to optimize (excluding the model's own parameters)
    params_to_optimize = [eigenmodel.u]
    #    param for name, param in eigenmodel.named_parameters() if name=='u'
    #]

    optimizer: Optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

    eigenmodel.normalize_parameters()

    for epoch in range(n_epochs):
      basis_losses: float = 0.0
      high_H_losses: float = 0.0
      total_losses: float = 0.0
      n_batches: int = 0
      u_dataloader = DataLoader(eigenmodel.u, batch_size=u_batch_size, shuffle=False)


      for x in x_dataloader:
        dH_du_list = []
        u_list = []
        n_batches += 1

        for u in u_dataloader:


          # Forward pass
          dH_du = eigenmodel(x.to(device), u)
            
          high_H_loss = -1*(dH_du**2).mean()*u.shape[0]/eigenmodel.u.shape[0]#/x.numel()
      
      
          high_H_loss.backward() # Step backward


          dH_du_list.append(dH_du.detach())
          u_list.append(u)
 

          high_H_losses = high_H_losses + high_H_loss.detach()


        # Normalize parameters at the start of each batch
        #eigenmodel.normalize_parameters()

        #optimizer.zero_grad()  # Clear gradients
        
        optimizer.step()
        optimizer.zero_grad()
        
        eigenmodel.normalize_parameters()
      
        dH_du_tensor = torch.concat(dH_du_list, dim=0)
        u_tensor = torch.concat(u_list, dim=0)
        cosine_sims = u_tensor @ u_tensor.transpose(0,1)
        prod = einops.einsum(dH_du_tensor, dH_du_tensor, 'k1 ... , k2 ...->... k1 k2')
        prod_cosin_sims = einops.einsum(prod, cosine_sims, '... k1 k2, k1 k2 ->... k1 k2')
        #pk = prod_cosin_sims[:,:,lower_triangular_mask]

        # Apply the mask to the last 2 dimensions of prod_cosin_sims
        basis_loss = lambda_penalty *  prod_cosin_sims[:,:,lower_triangular_mask].norm()/x.numel()
      
  
        (basis_loss).backward()
        
        optimizer.step()
        basis_losses = basis_losses + basis_loss.detach()
        optimizer.zero_grad()
        eigenmodel.normalize_parameters()
        torch.cuda.empty_cache()
        gc.collect()
      # Logging progress every 1% of total epochs
      if epoch % max(1, round(n_epochs / 100)) == 0:
        avg_total_loss = (high_H_losses + basis_losses) / n_batches
        avg_high_H_loss = high_H_losses / n_batches
        avg_basis_loss = basis_losses / n_batches
        print(
            f'Epoch {epoch} : {avg_total_loss:.3f},  '
            f'High Hessian Loss: {avg_high_H_loss:.3f},  '
            f'Basis Loss: {avg_basis_loss:.3f}'
              )