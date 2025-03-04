# eigenestimation.py
import torch
import torch.nn as nn
import einops
from torch.func import functional_call
from typing import Callable
from torch import Tensor
import copy

class EigenModel(nn.Module):
    def __init__(self, 
                 model: nn.Module, 
                 model0: Callable,
                 loss: Callable, 
                 n_features: int,
                 reduced_dim: int = 1,
                 device='cuda') -> None:
        super(EigenModel, self).__init__()

        self.model: nn.Module = model
        self.model0: Callable = model0
        self.loss: Callable = loss
        self.n_features = n_features
        self.param_dict = {name: param.detach().clone() for name, param in model.named_parameters()}
    
        print("HERE!")
        
        self.low_rank_decode = {name: [(torch.randn(length, reduced_dim, n_features)/n_features).to(device).requires_grad_(True) for length in param.shape]
                         for name, param in self.model.named_parameters()}
        self.normalize_low_ranks()
        #self.low_rank_encode = {name: [(torch.randn(length, reduced_dim, #n_features)/n_features).to(device).requires_grad_(True) for length in #param.shape]
        #                 for name, param in self.model.named_parameters()}
        self.low_rank_encode = copy.deepcopy(self.low_rank_decode)


    def compute_norm(self, network):
        return sum([(v**2).sum()**.5 for v in network.values()])**.5
    
    
    def normalize_low_ranks(self, eps=1e-10):
        #norm_current = self.compute_norms(self.reconstruct_network())
        #norm_goal = self.compute_norm(self.param_dict)
        norms = [self.compute_norm(network) for network in self.construct_subnetworks()]
        
        for name in self.low_rank_decode:
            n_low_rank_adaptors = len(self.low_rank_decode[name])
            for i in range(self.n_features):
                for d in range(n_low_rank_adaptors):
                    self.low_rank_decode[name][d][...,i].data.div_(
                        norms[i]**(1/n_low_rank_adaptors)+ eps)
                
    def compute_loss(self, x: torch.Tensor, param_dict) -> torch.Tensor:
        outputs: torch.Tensor = functional_call(self.model, param_dict, (x,))
        with torch.no_grad():
            truth: torch.Tensor = self.model0(outputs)
        return self.loss(outputs, truth)

    def compute_gradients(self, x: torch.Tensor,chunk_size=None):
        with torch.no_grad():
            grads = torch.func.jacrev(self.compute_loss, argnums=-1, has_aux=False, chunk_size=chunk_size)(x, self.param_dict)
            torch.cuda.empty_cache()
            return grads
    
    def forward(self, gradients: torch.Tensor) -> torch.Tensor:
        jvp_dict = dict({})
        for name in self.low_rank_encode:
            jvp_dict[name] = einops.einsum(gradients[name], self.low_rank_encode[name][-1], '... w , w r f -> ... r f')
            for tensor in self.low_rank_encode[name][-2::-1]:
                jvp_dict[name] = einops.einsum(jvp_dict[name], tensor, '... w r f, w r f -> ... r f')
            jvp_dict[name] = einops.einsum(jvp_dict[name], '... r f -> ... f')
        jvp = torch.stack([jvp_dict[name] for name in jvp_dict], dim=0).sum(dim=0) # Dimensions = (samples) x features
        return jvp
    
    def gradients_vector_product(self, gradients, feature_idx):
        jvp_dict = dict({})
        for name in self.low_rank_encode:
            jvp_dict[name] = einops.einsum(gradients[name], self.low_rank_encode[name][-1][...,feature_idx], '... w , w r-> r ...')
            for tensor in self.low_rank_encode[name][-2::-1]:
                jvp_dict[name] = einops.einsum(jvp_dict[name], tensor[...,feature_idx], 'r ... w, w r -> r ...')
            jvp_dict[name] = einops.einsum(jvp_dict[name], 'r ... w -> ...')

        jvp = torch.stack([jvp_dict[name] for name in jvp_dict], dim=0).sum(dim=0) # Dimensions = (samples) x features
        return jvp
    
    def reconstruct(self, jvp: torch.Tensor) -> torch.Tensor:
        reconstruction = dict({})
        for name in self.low_rank_decode:
            reconstruction[name] = einops.einsum(jvp, self.low_rank_decode[name][0], '... f , w r f -> ... w r f')
            for tensor in self.low_rank_decode[name][1:]:
                reconstruction[name] = einops.einsum(reconstruction[name], tensor, '... r f, w r f -> ... w r f')
            reconstruction[name] = einops.einsum(reconstruction[name], '... w r f -> ... w')
        return reconstruction


    def reconstruct_network(self) -> dict:
        reconstruction = dict({})
        for name in self.low_rank_decode:
            reconstruction[name] = self.low_rank_decode[name][0]
            for tensor in self.low_rank_decode[name][1:]:
                reconstruction[name] = einops.einsum(reconstruction[name], tensor, '... r f, w r f ->  ... w r f')
            reconstruction[name] = einops.einsum(reconstruction[name], '... w r f -> ... w')
        return reconstruction
    

    def add_to_network(self, feature_coefficients: torch.Tensor) -> dict:
        reconstruction = dict({})
        for name in self.low_rank_decode:
            reconstruction[name] = self.low_rank_decode[name][0]
            for tensor in self.low_rank_decode[name][1:]:
                reconstruction[name] = einops.einsum(reconstruction[name], tensor, '... r f, w r f ->  ... w r f')
            reconstruction[name] = einops.einsum(reconstruction[name], feature_coefficients, '... w r f, f -> ... w')
            reconstruction[name] = reconstruction[name] + self.param_dict[name]
        return reconstruction
    
    def construct_subnetworks(self) -> dict:
        networks = []
        for i in range(self.n_features):
            reconstruction = dict({})
            for name in self.low_rank_decode:
                reconstruction[name] = self.low_rank_decode[name][0][...,i]
                for tensor in self.low_rank_decode[name][1:]:
                    reconstruction[name] = einops.einsum(reconstruction[name], tensor[...,i], '... r, w r -> ... w r')
                reconstruction[name] = einops.einsum(reconstruction[name], '... w r -> ... w')
            networks.append(reconstruction)
        return networks
    
    