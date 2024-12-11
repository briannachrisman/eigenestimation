#@title Eigenestimation.py

import torch
import torch.nn as nn
from torch.nn.utils import stateless
from torch.func import jacrev, functional_call, jvp, vmap, jacrev
import einops
from typing import Any, Dict, List, Tuple, Callable
import gc
from functools import partial 
from torch.autograd.functional import jacobian

class EigenEstimationComparison(nn.Module):
    def __init__(self, 
    model: nn.Module, 
    model0: Callable,
    loss: Callable, 
    n_u_vectors: int, 
    u_chunk_size=10) -> None:
        super(EigenEstimationComparison, self).__init__()

        self.model: nn.Module = model
        self.model0: Callable = model0
        self.loss: Callable = loss
        self.n_u_vectors: int = n_u_vectors
        self.param_dict = {name: param.detach().clone() for name, param in model.named_parameters()}
        self.n_params = sum([v.numel() for v in self.param_dict.values()])
        self.u = nn.Parameter(torch.randn(n_u_vectors, self.n_params).requires_grad_(True))
        
    def params_to_vectors(self, param_dict):
        return torch.cat([param.flatten(-len(self.param_dict[name].shape),-1) for name, param in  param_dict.items()], dim=-1)
        
    # Restore parameters from a vector to the dictionary format
    def vector_to_parameters(self, vector):
      # Create an iterator to slice vector based on parameter shapes
      pointer = 0
      new_params = {}
        
      for name, param in self.param_dict.items():
        numel = param.numel()  # Number of elements in this parameter
        # Slice out `numel` elements from the vector
        new_params[name] = vector[pointer:pointer + numel].view(param.shape)
        pointer += numel
      return new_params
    


    def compute_loss(
        self, x: torch.Tensor, param_dict
    ) -> torch.Tensor:
        print('here')
        # Perform a stateless functional call to the model with given parameters
        #param_dict = self.vector_to_parameters(parameters)
        outputs: torch.Tensor = functional_call(self.model, param_dict, (x,))
       
        # Detach outputs to prevent gradients flowing back
        with torch.no_grad():
            truth: torch.Tensor = self.model(x)

        # CrossEntropyLoss needs to be of form (_, n_classes, ...)        
        #outputs = einops.rearrange(outputs, '... c -> c ...').unsqueeze(0)
        #truth = einops.rearrange(truth, '... c -> c ...').unsqueeze(0)

        # Compute the loss without reduction
        return self.loss(outputs, truth)#.squeeze(0)



    def normalize_parameters(self, eps=1e-6) -> None:
      # Concatenate all parameters into a single tensor
      with torch.no_grad():
        self.u.div_(eps+self.u.norm(keepdim=True, dim=1).detach())

    def forward(self, x: torch.Tensor, jacobian: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute the double gradient along u
        j_u = einops.einsum(jacobian, self.u, '... w, v w -> v ...')
        return j_u #, einops.einsum(j_u, j_u, 'v1 ..., v2 ...  -> v1 v2 ...')