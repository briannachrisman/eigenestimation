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
        self.n_features = n_features
        self.param_dict = {name: param.detach().clone() for name, param in model.named_parameters()}
        self.low_rank = {name: [(torch.randn(length, reduced_dim, n_features)/n_features).to(device).requires_grad_(True) for length in param.shape]
                         for name, param in self.model.named_parameters()}

    def compute_loss(self, x: torch.Tensor, param_dict) -> torch.Tensor:
        outputs: torch.Tensor = functional_call(self.model, param_dict, (x,))
        with torch.no_grad():
            truth: torch.Tensor = self.model0(outputs)
        return self.loss(outputs, truth)

    def compute_jacobian(self, x: torch.Tensor):
        return torch.func.jacrev(self.compute_loss, argnums=-1)(x, self.param_dict)
    
    def forward(self, jacobian: torch.Tensor) -> torch.Tensor:
        jvp_dict = dict({})
        for name in self.low_rank:
            jvp_dict[name] = einops.einsum(jacobian[name], self.low_rank[name][-1], '... w , w r f -> f r ...')
            for tensor in self.low_rank[name][-2:0:-1]:
                jvp_dict[name] = einops.einsum(jvp_dict[name], tensor, 'f r ... w, w r f -> f r ...')
            jvp_dict[name] = einops.einsum(jvp_dict[name], self.low_rank[name][0], 'f r ... w, w r f -> ... f')
        jvp = torch.stack([jvp_dict[name] for name in jvp_dict], dim=0).sum(dim=0) # Dimensions = (samples) x features
        return jvp
    
    def jacobian_vector_product(self, jacobian, feature_idx):
        jvp_dict = dict({})
        for name in self.low_rank:
            jvp_dict[name] = einops.einsum(jacobian[name], self.low_rank[name][-1][...,feature_idx], '... w , w r-> r ...')
            for tensor in self.low_rank[name][-2:0:-1]:
                jvp_dict[name] = einops.einsum(jvp_dict[name], tensor[...,feature_idx], 'r ... w, w r -> r ...')
            jvp_dict[name] = einops.einsum(jvp_dict[name], self.low_rank[name][0][...,feature_idx], 'r ... w, w r-> ...')
        jvp = torch.stack([jvp_dict[name] for name in jvp_dict], dim=0).sum(dim=0) # Dimensions = (samples) x features
        return jvp
    
    def reconstruct(self, jvp: torch.Tensor) -> torch.Tensor:
        reconstruction = dict({})
        for name in self.low_rank:
            reconstruction[name] = einops.einsum(jvp, self.low_rank[name][0], '... f , w r f -> ... w f r')
            for tensor in self.low_rank[name][1:-1]:
                reconstruction[name] = einops.einsum(reconstruction[name], tensor, '... f r, w r f -> ... w f r')
            reconstruction[name] = einops.einsum(reconstruction[name], self.low_rank[name][-1], '... f r, w r f -> ... w')
        return reconstruction


    def reconstruct_network(self) -> dict:
        reconstruction = dict({})
        for name in self.low_rank:
            reconstruction[name] = self.low_rank[name][0]
            for tensor in self.low_rank[name][1:-1]:
                reconstruction[name] = einops.einsum(reconstruction[name], tensor, 'r f, w r f ->  w r f')
            reconstruction[name] = einops.einsum(reconstruction[name], self.low_rank[name][-1], '... r f, w r f -> ... w')
        return reconstruction
    
    def construct_subnetworks(self) -> dict:
        networks = []
        for i in range(self.n_features):
            reconstruction = dict({})
            for name in self.low_rank:
                reconstruction[name] = self.low_rank[name][0][...,i]
                for tensor in self.low_rank[name][1:-1]:
                    reconstruction[name] = einops.einsum(reconstruction[name], tensor[...,i], '... r, w r -> ... w r')
                reconstruction[name] = einops.einsum(reconstruction[name], self.low_rank[name][-1][...,i], '... r, w r-> ... w')
            networks.append(reconstruction)
        return networks