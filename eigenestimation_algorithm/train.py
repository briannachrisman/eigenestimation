import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Any, List
import einops
import gc
import math
import time 
def TrainEigenEstimation(
    eigenmodel: nn.Module,
    x_dataloader: DataLoader,
    lr: float,
    n_epochs: int,
    lambda_penalty: List[float],
    u_batch_size = 16,
    device: str = 'cuda'
) -> None:

    t = time.time()
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
      #idx_dataloader = DataLoader(torch.arange(len(eigenmodel.u)), batch_size=u_batch_size, shuffle=True)


      for x in x_dataloader:
        n_batches += 1

        if True:
          #print('1', t-time.time())

          # Create a lower triangular mask for loss computation
          lower_triangular_mask: torch.Tensor = torch.tril(
              torch.ones(eigenmodel.u.shape[0],eigenmodel.u.shape[0], dtype=torch.bool), diagonal=-1
          ).to(device)
          dH_du_list = []
          #print('2', t-time.time())
          u_list = []
          dP_du, FIM_diag= eigenmodel(x, u) # n_classes n_u
          
          FIM_diag = einops.einsum(dP_du, dP_du, 'v1 ... c, v2 ... c -> v1 v2 ...') #/  dP_du.shape[-1]
          H = einops.einsum(dP_du, dP_du, 'v1 ... p, v2 ... p -> v1 v2 ...')
          lower_triangular_mask: torch.Tensor = torch.tril(
              torch.ones(H.shape[0],H.shape[0], dtype=torch.bool), diagonal=-1
              ).to(device)


          diag: torch.Tensor = torch.eye(H.shape[0], dtype=torch.bool).to(device)
          not_diag = (1-diag.float()).bool()

          ##print(FIM_diag.shape, dP_du.shape)
          
          FIM2_loss = (H[diag,...]).mean()#-(FIM_diag.mean())
          FIM_cosine_sim_loss = (H[not_diag,...]**2).mean()#/2#FIM2_flat = einops.rearrange(FIM_diag, 'v n ... -> (n ...) v')
          


          L = -FIM2_loss + lambda_penalty[0] * FIM_cosine_sim_loss + lambda_penalty[1]*abs(eigenmodel.u).mean()
          L.backward()
          optimizer.step()
          optimizer.zero_grad()
          ##print(eigenmodel, eigenmodel.u, 'pre] norm')

          eigenmodel.normalize_parameters()
          ##print(eigenmodel, eigenmodel.u, 'post norm')
          total_losses = total_losses + L.detach()
          high_H_losses = high_H_losses + FIM2_loss.detach()
          basis_losses = basis_losses + FIM_cosine_sim_loss.detach()         
          
          #torch.cuda.empty_cache()
          #gc.collect()
          break
      # Logging progress every 1% of total epochs
      if epoch % max(1, round(n_epochs / 100)) == 0:
        avg_total_loss =total_losses / n_batches
        avg_high_H_loss = high_H_losses / n_batches
        avg_basis_loss = basis_losses / n_batches
        print(
            f'Epoch {epoch} : {avg_total_loss:.3f},  '
            f'High Hessian Loss: {avg_high_H_loss:.3f},  '
            f'Basis Loss: {avg_basis_loss:.3f}'
              )

def TrainEigenEstimationComparison(
    eigenmodel: nn.Module,
    x_dataloader: DataLoader,
    lr: float,
    n_epochs: int,
    lambda_penalty: List[float],
    u_batch_size = 16,
    jac_chunk_size = None,
    device: str = 'cuda'
) -> None:

  


    # Collect parameters to optimize (excluding the model's own parameters)
    params_to_optimize = [eigenmodel.u]
    #    param for name, param in eigenmodel.named_parameters() if name=='u'
    #]

    optimizer: Optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    eigenmodel.normalize_parameters()

    for epoch in range(n_epochs):
      t = time.time()
      basis_losses: float = 0.0
      high_H_losses: float = 0.0
      total_losses: float = 0.0
      n_batches: int = 0
      #idx_dataloader = DataLoader(torch.arange(len(eigenmodel.u)), batch_size=u_batch_size, shuffle=True)

      #print('1', t-time.time())

      for x,jac in x_dataloader:
        
        n_batches += 1
        #print('1', t-time.time())

        if True:
          dH_du_list = []
          u_list = []
          ##print('2', t-time.time())

          dP_du = eigenmodel(x, jac) # n_classes n_u
          #print('3', t-time.time())

          H = einops.einsum(dP_du, dP_du, 'v1 ..., v2 ... -> v1 v2 ...') #/  dP_du.shape[-1]
          #print('4', t-time.time())
          lower_triangular_mask: torch.Tensor = torch.tril(
              torch.ones(H.shape[0],H.shape[0], dtype=torch.bool), diagonal=-1
              ).to(device)

          #print('4', t-time.time())
          diag: torch.Tensor = torch.eye(H.shape[0], dtype=torch.bool).to(device)
          not_diag = (1-diag.float()).bool()

          ##print(FIM_diag.shape, dP_du.shape)
          #print('5', t-time.time())

          n_samples = math.prod(list(H.shape[2:]))
          FIM2_loss = dP_du.relu().sum()/dP_du.shape[0]/n_samples#-(FIM_diag.mean())
          FIM_cosine_sim_loss = abs(H[not_diag,...]).sum()/dP_du.shape[0]/n_samples #*H.shape[0]#/sum(H.shape[2:])#/2#FIM2_flat = einops.rearrange(FIM_diag, 'v n ... -> (n ...) v')

          #print('6', t-time.time())

          L = -FIM2_loss + lambda_penalty[0] * FIM_cosine_sim_loss + lambda_penalty[1]*abs(eigenmodel.u).mean()
          L.backward()
          optimizer.step()
          optimizer.zero_grad()
          #print('7', t-time.time())

          ##print(eigenmodel, eigenmodel.u, 'pre] norm')

          eigenmodel.normalize_parameters()
          #print('8', t-time.time())
          ##print(eigenmodel, eigenmodel.u, 'post norm')
          total_losses = total_losses + L.detach()
          high_H_losses = high_H_losses + FIM2_loss.detach()
          basis_losses = basis_losses + FIM_cosine_sim_loss.detach()         
          
          #torch.cuda.empty_cache()
          #gc.collect()
          
      # Logging progress every 10% of total epochs
      if (epoch+1) % max(1, round(n_epochs / 10)) == 0:
        avg_total_loss =total_losses / n_batches
        avg_high_H_loss = high_H_losses / n_batches
        avg_basis_loss = basis_losses / n_batches
        print(
          f'Epoch {epoch} : {avg_total_loss:.3f},  '
          f'High Hessian Loss: {avg_high_H_loss:.3f},  '
          f'Basis Loss: {avg_basis_loss:.3f}'
        )