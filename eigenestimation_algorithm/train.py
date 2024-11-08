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
      #u_dataloader = DataLoader(eigenmodel.u, batch_size=u_batch_size, shuffle=False)
      #idx_dataloader = DataLoader(torch.arange(len(eigenmodel.u)), batch_size=u_batch_size, shuffle=True)


      for x in x_dataloader:
        #for idx in idx_dataloader:
          dH_du_list = []
          u_list = []
          n_batches += 1

          dP_du = eigenmodel(x.to(device), eigenmodel.u)

          FIM_diag = einops.einsum(dP_du, dP_du, 'v ... c, v ... c -> v ...') #/  dP_du.shape[-1]
          FIM2_loss = -(FIM_diag**2).mean()
          FIM2_flat = einops.rearrange(FIM_diag, 'v n ... -> v (n ...)')
          u_cosine_samples = abs(einops.einsum(FIM2_flat, FIM2_flat, 'v1 n,v2 n->v1 v2'))/FIM2_flat.shape[-1]
          FIM_cosine_sim_loss = u_cosine_samples[lower_triangular_mask].mean()
          
          #dP_du = eigenmodel_transformer(X_transformer[:4,:16].to(device), eigenmodel_transformer.u)
          #lower_triangular_mask: torch.Tensor = torch.tril(
          #        torch.ones(dP_du.shape[1], dP_du.shape[1], dtype=torch.bool), diagonal=-1
          #).to(device)
          #FIM_diag = einops.einsum(dP_du, dP_du, 'v ... c, v ... c -> v ...') #/  dP_du.shape[-1]
          #FIM2_loss = -(FIM_diag**2).mean()
          #u_cosine_each_samples = abs(einops.einsum(FIM_diag, FIM_diag, 'v b1 t1 ,v b2 t2 -> b1 b2 t1 t2 '))#/FIM2_flat.shape[-1]
          #          FIM2_flat = einops.rearrange(FIM_diag, 'v n ... -> v (n ...)')
          #          u_cosine_samples = einops.einsum(FIM2_flat, FIM2_flat, 'v1 n,v2 n->v1 v2')/FIM2_flat.shape[-1]
          #FIM_cosine_sim_loss = u_cosine_each_samples[lower_triangular_mask,:,:].mean()

          #high_H_loss = -1*(dH_du**2).mean()*u.shape[0]/eigenmodel.u.shape[0]#/x.numel()
        
        
          #  high_H_loss.backward() # Step backward


          #  dH_du_list.append(dH_du.detach())
          #  u_list.append(u)
  



          # Normalize parameters at the start of each batch
          #eigenmodel.normalize_parameters()
    
          #optimizer.zero_grad()  # Clear gradients
          
          
          #eigenmodel.normalize_parameters()
        
          #dH_du_tensor = torch.concat(dH_du_list, dim=0)
          #u_tensor = torch.concat(u_list, dim=0)
          #cosine_sims = u_tensor @ u_tensor.transpose(0,1)
          #prod = einops.einsum(dH_du_tensor, dH_du_tensor, 'k1 ... , k2 ...->... k1 k2')
          #prod_cosin_sims = einops.einsum(prod, cosine_sims, '... k1 k2, k1 k2 ->... k1 k2')
          #pk = prod_cosin_sims[:,:,lower_triangular_mask]

          # Apply the mask to the last 2 dimensions of prod_cosin_sims
          #basis_loss = lambda_penalty *  prod_cosin_sims[:,:,lower_triangular_mask].norm()/x.numel()
        
    


          L = FIM2_loss + lambda_penalty * FIM_cosine_sim_loss
          L.backward()
          optimizer.step()
          optimizer.zero_grad()
          eigenmodel.normalize_parameters()

          high_H_losses = high_H_losses + FIM2_loss.detach()
          basis_losses = basis_losses + FIM_cosine_sim_loss.detach()         
          
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