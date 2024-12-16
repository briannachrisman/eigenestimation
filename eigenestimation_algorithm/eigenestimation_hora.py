import torch
import torch.nn as nn
from torch.nn.utils import stateless
from torch.func import jacrev, functional_call, jvp, vmap, jacrev
import einops
from typing import Any, Dict, List, Tuple, Callable
import gc
from functools import partial 
from torch.autograd.functional import jacobian
import einops
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Any, List
import einops
import gc
import math
import time 
from torch import Tensor

def HoraShape(tensor, n_dim, n_features):
    return einops.repeat(
        einops.einsum(
            tensor, 'h w ... -> h ...'), 'h ... -> h d ... r', r=n_features, d=n_dim).shape

class EigenHora(nn.Module):
    def __init__(self, 
    model: nn.Module, 
    model0: Callable,
    loss: Callable, 
    n_features: int,
    reduced_dim: int =1,
    device='cuda') -> None:
        super(EigenHora, self).__init__()

        self.model: nn.Module = model
        self.model0: Callable = model0
        self.loss: Callable = loss
        self.param_dict = {name: param.detach().clone() for name, param in model.named_parameters()}
        self.n_params = sum([v.numel() for v in self.param_dict.values()])
        self.u_left = {name:  
            (torch.randn(HoraShape(param, reduced_dim, n_features))
             ).to(device).requires_grad_(True) for name, param in self.model.named_parameters()}
        self.u_right = {name:  
            (torch.randn(HoraShape(param.transpose(0,1), reduced_dim, n_features))/n_features
             ).to(device).requires_grad_(True) for name, param in self.model.named_parameters()}
        



    def compute_loss(
        self, x: torch.Tensor, param_dict
    ) -> torch.Tensor:
        # Perform a stateless functional call to the model with given parameters
        #param_dict = self.vector_to_parameters(parameters)
        outputs: torch.Tensor = functional_call(self.model, param_dict, (x,))
       
        # Detach outputs to prevent gradients flowing back
        with torch.no_grad():
            truth: torch.Tensor = self.model0(x)
            
        # Compute the loss without reduction
        return einops.einsum(self.loss(outputs, truth), 's ... -> s')#.squeeze(0)

    def compute_jacobian(self, x:torch.Tensor):
        return torch.func.jacrev(self.compute_loss, argnums=-1)(x, self.param_dict)
    
    def forward(self, jacobian: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #3print(jacobian['W_in'].shape)
        #print(self.u_right['W_in'].shape)
        jac_left = {name:einops.einsum(jacobian[name], self.u_left[name], 's h w ... , h d ... f -> s ... d w f') for name in jacobian}
        jvp_dict = {name:einops.einsum(jac_left[name], self.u_right[name], 's ... d w f, w d ... f -> s ... f') for name in jac_left}
        jvp = einops.einsum(torch.stack([jvp_dict[name] for name in jvp_dict], dim=0).sum(dim=0), 's ... f -> s f')
        return jvp
    
    
    def reconstruct(self, jvp: torch.Tensor)-> torch.Tensor:
        reconstruction_left = {name:einops.einsum(jvp, self.u_left[name], 's f, h d ... f ->  s h d  ... f') for name in self.u_left}
        reconstruction = {name:einops.einsum(reconstruction_left[name], self.u_right[name], 's h d ... f, w d ... f -> s ... h w') for name in reconstruction_left}
        return reconstruction
    
    
eps = 1e-10
def Train(
    eigenmodel: nn.Module,
    jacobian_dataloader: DataLoader,
    lr: float,
    n_epochs: int,
    L0_penalty: float,
    device: str = 'cuda'
) -> None:

    # Collect parameters to optimize (excluding the model's own parameters)
    params_to_optimize = [*[eigenmodel.u_left[name] for name in eigenmodel.u_left], *[eigenmodel.u_right[name] for name in eigenmodel.u_right]]
    optimizer: Optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    
    for epoch in range(n_epochs):
        sparsity_losses: float = 0.0
        reconstruction_losses: float = 0.0
        total_losses: float = 0.0
        n_batches: int = 0
        for jacobian in jacobian_dataloader:
            n_batches = n_batches + 1
            jvp = eigenmodel(jacobian)
            reconstruction = eigenmodel.reconstruct(jvp.relu())
            L2_error = torch.stack([einops.einsum(
                (reconstruction[name] - jacobian[name])**2, 's ... h w -> s') for name in jacobian]).mean()
            L0_error = einops.einsum((abs(jvp)+ eps), 's ... f -> s').mean()
            L = L2_error + L0_penalty * L0_error
            
            sparsity_losses = sparsity_losses + L0_error
            reconstruction_losses = reconstruction_losses + L2_error
            total_losses = total_losses + L
        
            L.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Logging progress every 1% of total epochs
        if epoch % max(1, round(n_epochs / 100)) == 0:
            avg_total_loss =total_losses / n_batches
            avg_sparsity_loss = sparsity_losses / n_batches
            avg_reconstruction_loss = reconstruction_losses / n_batches
            print(
                f'Epoch {epoch} : {avg_total_loss:.3f},  '
                f'Reconstruction Loss: {avg_reconstruction_loss:.3f},  '
                f'Sparsity Loss: {avg_sparsity_loss:.3f}'
                )
            