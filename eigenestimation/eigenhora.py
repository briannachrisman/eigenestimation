# eigenestimation.py
import torch
import torch.nn as nn
import einops
from torch.func import functional_call
from typing import Callable
from torch import Tensor
from eigenestimation.utils import HoraShape

class EigenHora(nn.Module):
    def __init__(self, 
                 model: nn.Module, 
                 model0: Callable,
                 loss: Callable, 
                 n_features: int,
                 reduced_dim: int = 1,
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
            (torch.randn(HoraShape(param.transpose(0, 1), reduced_dim, n_features))/n_features
             ).to(device).requires_grad_(True) for name, param in self.model.named_parameters()}

    def compute_loss(self, x: torch.Tensor, param_dict) -> torch.Tensor:
        outputs: torch.Tensor = functional_call(self.model, param_dict, (x,))
        with torch.no_grad():
            truth: torch.Tensor = self.model0(x)
        return einops.einsum(self.loss(outputs, truth), 's ... -> s')

    def compute_jacobian(self, x: torch.Tensor):
        return torch.func.jacrev(self.compute_loss, argnums=-1)(x, self.param_dict)
    
    def forward(self, jacobian: torch.Tensor) -> torch.Tensor:
        jac_left = {name: einops.einsum(jacobian[name], self.u_left[name], 's h w ... , h d ... f -> s ... d w f') for name in jacobian}
        jvp_dict = {name: einops.einsum(jac_left[name], self.u_right[name], 's ... d w f, w d ... f -> s ... f') for name in jac_left}
        jvp = einops.einsum(torch.stack([jvp_dict[name] for name in jvp_dict], dim=0).sum(dim=0), 's ... f -> s f')
        return jvp
    
    def reconstruct(self, jvp: torch.Tensor) -> torch.Tensor:
        reconstruction_left = {name: einops.einsum(jvp, self.u_left[name], 's f, h d ... f -> s h d  ... f') for name in self.u_left}
        reconstruction = {name: einops.einsum(reconstruction_left[name], self.u_right[name], 's h d ... f, w d ... f -> s ... h w') for name in reconstruction_left}
        return reconstruction
