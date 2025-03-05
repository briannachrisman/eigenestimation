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
        jvp = None
        for name in self.low_rank_encode:
            rank = self.low_rank_encode[name][-1][...,-1].shape[-1]
            for r in range(rank):
                partial_tmp = einops.einsum(gradients[name], self.low_rank_encode[name][-1][:,r,:], '... w , w f -> ... f')
                for tensor in self.low_rank_encode[name][-2::-1]:
                    partial_tmp = einops.einsum(partial_tmp, tensor[...,r,:], '... w f, w f -> ... f')
                    if jvp is None:
                        jvp = partial_tmp
                    else:
                        jvp += partial_tmp
        return jvp
    
    

    
    def reconstruct(self, jvp: torch.Tensor) -> torch.Tensor:
        reconstruction = {}
        for name, tensors in self.low_rank_decode.items():
            temp = None

            # Iterate over feature dimension f to perform contractions in a memory-efficient way
            for f in range(self.n_features):
                for r in range(tensors[0][...,-1].shape[-1]):
                    partial_temp = einops.einsum(jvp[...,f], tensors[0][..., r,f], '... , w -> ... w')
                    for tensor in tensors[1:]:
                        partial_temp = einops.einsum(partial_temp, tensor[..., r,f], '..., w -> ... w')
                    if temp is None:
                        temp = partial_temp
                    else: 
                        temp += partial_temp  # Accumulate over f instead of processing all at once

            reconstruction[name] = temp # Final reduction
        return reconstruction

    


    def reconstruct_network(self) -> dict:
        reconstruction = {}
        for name, tensors in self.low_rank_decode.items():
            temp = None

            # Iterate over low-rank dimension r
            for r in range(tensors[0].shape[-2]):
                partial_temp = None

                # Iterate over feature dimension f
                for f in range(tensors[0].shape[-1]):
                    intermediate = tensors[0][:, r, f]  # Extract single element from the first tensor
                    
                    # Multiply through the remaining tensors
                    for tensor in tensors[1:]:
                        intermediate = einops.einsum(intermediate, tensor[r, :, f], '... w, w -> ... w')

                    # Accumulate over f
                    if partial_temp is None:
                        partial_temp = intermediate
                    else:
                        partial_temp += intermediate

                # Accumulate over r
                if temp is None:
                    temp = partial_temp
                else:
                    temp += partial_temp  

            reconstruction[name] = temp  # No need for a final einsum since both `r` and `f` are summed out

        return reconstruction

    

    def add_to_network(self, feature_coefficients: dict) -> dict:
        reconstruction = copy.deepcopy(self.param_dict)
        for name in self.low_rank_decode:
            for feature_idx in feature_coefficients:
                sum_so_far = None
                for r in range(self.low_rank_decode[name][0].shape[-2]):
                    tmp = self.low_rank_decode[name][0][...,r,feature_idx]
                    for tensor in self.low_rank_decode[name][1:]:
                        tmp = einops.einsum(tmp, tensor[...,r,feature_idx], '..., w -> ... w')
                        if sum_so_far is None:
                            sum_so_far = tmp
                        else:
                            sum_so_far += tmp
                reconstruction[name] = reconstruction[name] + feature_coefficients[feature_idx] * sum_so_far
        return reconstruction
    
    def construct_subnetworks(self) -> dict:
        networks = []
        for i in range(self.n_features):
            reconstruction = dict({})
            for name in self.low_rank_decode:
                for r in range(self.low_rank_decode[name][0].shape[-2]):
                    sum_so_far = None
                    tmp = self.low_rank_decode[name][0][...,r,i]
                    for tensor in self.low_rank_decode[name][1:]:
                        tmp = einops.einsum(tmp, tensor[...,r,i], '..., w -> ... w')
                        if sum_so_far is None:
                            sum_so_far = tmp
                        else:
                            sum_so_far += tmp
                reconstruction[name] = sum_so_far
            networks.append(reconstruction)
        return networks
    
    