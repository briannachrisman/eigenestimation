#@title Eigenestimation.py

import torch
import torch.nn as nn
from torch.nn.utils import stateless
from torch.func import jacrev, functional_call, jvp, vmap, jacrev
import einops
from typing import Any, Dict, List, Tuple, Callable
import gc
from functools import partial 

class EigenEstimation(nn.Module):
    def __init__(self, model: nn.Module, loss: Callable, n_u_vectors: int, u_chunk_size=10) -> None:
        super(EigenEstimation, self).__init__()

        self.model: nn.Module = model
        self.loss: Callable = loss
        self.n_u_vectors: int = n_u_vectors
        self.named_parameters = {name: param.detach().clone() for name, param in model.named_parameters()}
        self.w0 = self.parameters_to_vector(self.named_parameters)
        self.u = nn.Parameter(torch.randn((n_u_vectors, len(self.w0))).requires_grad_(True))
        self.u_chunk_size = u_chunk_size


        # Register u vectors as parameters with modified names
        #for name, tensor in u_dict.items():
        #    self.register_parameter(name.replace('.', '__'), tensor)

    def parameters_to_vector(self, named_parameters):
        return torch.cat([param.view(-1) for name, param in named_parameters.items()])

    # Restore parameters from a vector to the dictionary format
    def vector_to_parameters(self, vector):
      # Create an iterator to slice vector based on parameter shapes
      pointer = 0
      new_params = {}
        
      for name, param in self.named_parameters.items():
        numel = param.numel()  # Number of elements in this parameter
        # Slice out `numel` elements from the vector
        new_params[name] = vector[pointer:pointer + numel].view(param.shape)
        pointer += numel
      return new_params
    


    def compute_loss(
        self, x: torch.Tensor, parameters: torch.Tensor
    ) -> torch.Tensor:
        # Perform a stateless functional call to the model with given parameters
        param_dict = self.vector_to_parameters(parameters)
        outputs: torch.Tensor = functional_call(self.model, param_dict, (x,))
        # Detach outputs to prevent gradients flowing back
        with torch.no_grad():
            truth: torch.Tensor = self.model(x,)

        # CrossEntropyLoss needs to be of form (_, n_classes, ...)        
        #outputs = einops.rearrange(outputs, '... c -> c ...').unsqueeze(0)
        #truth = einops.rearrange(truth, '... c -> c ...').unsqueeze(0)

        # Compute the loss without reduction
        return self.loss(outputs, truth)#.squeeze(0)

    def double_grad_along_u(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:

        #return vmap(test, in_dims=(None, 0), out_dims=0, chunk_size=20)(X_transformer[:32], eigenmodel_transformer.u[:100])
        # Compute the first derivative along u.
        def inner_jvp(w0):
          return jvp(
              partial(self.compute_loss, x), (w0,), (u,)
              )[1]
              
        # Compute the second derivative along u.
        return inner_jvp(self.w0)#jvp(inner_jvp, (self.w0,), (u,))[1]

    def vmap_double_grad_along_u(self, x, us):
      #jr = jacrev(self.compute_loss, argnums=1)(x, self.w0)
      #print(jr)
      #return einops.einsum(jr, us, '... w, v w -> v ...')
      #return jvp(partial(self.compute_loss, x), (eigenmodel_transformer.w0,), (us,))[1]

      return vmap(self.double_grad_along_u, in_dims=(None, 0), out_dims=0, chunk_size=self.u_chunk_size)(x, us)

    def normalize_parameters(self, eps=1e-6) -> None:
      # Concatenate all parameters into a single tensor
      with torch.no_grad():
        self.u.div_(eps+self.u.norm(keepdim=True, dim=1).detach())


    def forward(self, x: torch.Tensor, parameters) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute the double gradient along u
        dH_du: torch.Tensor = self.vmap_double_grad_along_u(x, parameters)

        return dH_du, einops.einsum(dH_du, dH_du, 'v ... c, v ... c -> v ...')
        