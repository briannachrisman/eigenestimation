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

class JacobianExtractor(nn.Module):
    def __init__(self, model: nn.Module, model0: Callable, 
     param_dict,
     loss: Callable,
     chunk_size=10) -> None:
        super(JacobianExtractor, self).__init__()
        self.model = model
        self.model0 = model0
        self.loss = loss
        self.chunk_size = chunk_size
        self.param_dict = param_dict

    def params_to_vectors(self, param_dict):
        return torch.cat([param_dict[name].flatten(-len(self.param_dict[name].shape), -1) for name in param_dict], dim=-1)

    def compute_loss(self, x: torch.Tensor, param_dict) -> torch.Tensor:
        outputs = functional_call(self.model, param_dict, (x,))
        with torch.no_grad():
            truth = self.model0(x)
        loss = self.loss(outputs, truth)
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partial_func = partial(self.compute_loss, x)
        jac = jacrev(partial_func, chunk_size=self.chunk_size)(self.param_dict)
        print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e6} MB")
        print(f"Cached memory: {torch.cuda.memory_reserved() / 1e6} MB")
        j_vector = self.params_to_vectors(jac)
        return j_vector

def ExtractJacs(jac_extractor, x_dataloader, device='cuda'):
    jacs = []
    x_list = []
    for i, x_batch in enumerate(x_dataloader):
        print(f"Processing batch {i+1}/{len(x_dataloader)}")
        jac = jac_extractor(x_batch)
        jacs.append(jac.detach().cpu())
        x_list.append(x_batch.detach().cpu())
        del jac, x_batch
        torch.cuda.empty_cache()
        gc.collect()
    return torch.concat(jacs, dim=0), torch.concat(x_list, dim=0)
