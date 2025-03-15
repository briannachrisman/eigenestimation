# eigenestimation.py
import torch
import torch.nn as nn
import einops
from torch.func import functional_call
from typing import Callable
from torch import Tensor
import copy
import gc
from eigenestimation.utils.einsum_patterrns import generate_einsum_pattern, generate_forward_einsum_pattern, generate_reconstruct_einsum_pattern, generate_reconstruct_network_einsum_pattern, generate_add_to_network_einsum_pattern

class EigenModel(nn.Module):
    def __init__(self, 
                 model: nn.Module, 
                 model0: Callable,
                 loss: Callable, 
                 n_features: int,
                 reduced_dims: dict,
                 device='cuda') -> None:
        super(EigenModel, self).__init__()

        self.model: nn.Module = model
        self.model0: Callable = model0
        self.loss: Callable = loss
        self.n_features = n_features
        self.param_dict = {name: param.detach().clone() for name, param in model.named_parameters()}
        self.rank_dict = reduced_dims
        
        self.low_rank_decode = {name: 
            {
                'core_tensor': (torch.randn(*(self.rank_dict[name]), n_features)/n_features).to(device).requires_grad_(True),
                'transform_tensors':
                    [(torch.randn(self.rank_dict[name][d], self.param_dict[name].shape[d], n_features)/n_features).to(device).requires_grad_(True) for d in range(len(self.rank_dict[name]))]
            }
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
            n_low_rank_adaptors = len(self.low_rank_decode[name]['transform_tensors'])+1
            for i in range(self.n_features):
                self.low_rank_decode[name]['core_tensor'][...,i].data.div_(
                    norms[i]**(1/n_low_rank_adaptors)+ eps)
                for tensor in self.low_rank_decode[name]['transform_tensors']:
                    tensor[...,i].data.div_(
                        norms[i]**(1/n_low_rank_adaptors)+ eps)
        torch.cuda.empty_cache()
                
    def compute_loss(self, x: torch.Tensor, param_dict) -> torch.Tensor:
        self.model.eval()
        outputs: torch.Tensor = functional_call(self.model, param_dict, (x,))
        with torch.no_grad():
            truth: torch.Tensor = self.model0(outputs)
        torch.cuda.empty_cache()
        return self.loss(outputs, truth)

    def compute_gradients(self, x: torch.Tensor,chunk_size=None):
        with torch.no_grad():
            grads = torch.func.jacrev(self.compute_loss, argnums=-1, has_aux=False, chunk_size=chunk_size)(x, self.param_dict)
            torch.cuda.empty_cache()
            return grads


    def forward(self, gradients: torch.Tensor) -> torch.Tensor:
        jvp_dict = dict({})
        # Gradients # b d1 d2 d3...
        jvp = None
        for name in self.low_rank_encode:
            
            pattern = generate_forward_einsum_pattern(len(self.rank_dict[name]))
            jvp_so_far = einops.einsum(
                self.low_rank_encode[name]['core_tensor'], 
                *(self.low_rank_encode[name]['transform_tensors']), 
                gradients[name],
                pattern)
            if jvp is None:
                jvp = jvp_so_far
            else:
                jvp += jvp_so_far
        torch.cuda.empty_cache()
        return jvp
    

    def reconstruct(self, jvp: torch.Tensor) -> torch.Tensor:
        reconstruction = {}
        for name in self.low_rank_decode:
            pattern = generate_reconstruct_einsum_pattern(len(self.rank_dict[name]))
            reconstruction[name] = einops.einsum(jvp, self.low_rank_decode[name]['core_tensor'], *(self.low_rank_decode[name]['transform_tensors']), pattern)
            torch.cuda.empty_cache()
        return reconstruction


    def reconstruct_network(self) -> dict:
        reconstruction = {}
        for name in self.low_rank_decode:
            pattern = generate_reconstruct_network_einsum_pattern(len(self.rank_dict[name]))
            reconstruction[name] = einops.einsum(self.low_rank_decode[name]['core_tensor'], *(self.low_rank_decode[name]['transform_tensors']), pattern)
        torch.cuda.empty_cache()
        return reconstruction

    

    def add_to_network(self, feature_coefficients: torch.tensor) -> dict:
        reconstruction = {}
        for name in self.low_rank_decode:
            pattern = generate_add_to_network_einsum_pattern(len(self.rank_dict[name]))
            reconstruction[name] = self.param_dict[name] + einops.einsum(feature_coefficients, self.low_rank_decode[name]['core_tensor'], *(self.low_rank_decode[name]['transform_tensors']), pattern)
        torch.cuda.empty_cache()
        return reconstruction
    
    def construct_subnetworks(self) -> dict:
        networks = {}
        for name in self.low_rank_decode:
            pattern = generate_einsum_pattern(len(self.rank_dict[name]))
            networks[name] = einops.einsum(self.low_rank_decode[name]['core_tensor'], *(self.low_rank_decode[name]['transform_tensors']), pattern)
        torch.cuda.empty_cache()
        return [{name:networks[name][...,i] for name in networks} for i in range(self.n_features)]
    
    