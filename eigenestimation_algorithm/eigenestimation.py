import torch
import torch.nn as nn
from torch.nn.utils import stateless
from torch.func import jacrev, functional_call
import einops
from typing import Any, Dict, List, Tuple, Callable
import gc
class EigenEstimation(nn.Module):
    def __init__(self, model: nn.Module, loss: Callable, n_u_vectors: int) -> None:
        super(EigenEstimation, self).__init__()

        self.model: nn.Module = model
        self.loss: Callable = loss
        self.n_u_vectors: int = n_u_vectors
        self.w0: Dict[str, nn.Parameter] = dict(model.named_parameters())

        # Initialize u vectors as parameters
        u_dict: Dict[str, nn.Parameter] = {
            name: nn.Parameter(torch.stack([torch.randn_like(param) for _ in range(n_u_vectors)]))
            for name, param in self.w0.items()
        }

        # Register u vectors as parameters with modified names
        for name, tensor in u_dict.items():
            self.register_parameter(name.replace('.', '__'), tensor)

    def compute_loss(
        self, x: torch.Tensor, parameters: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Perform a stateless functional call to the model with given parameters
        outputs: torch.Tensor = functional_call(self.model, parameters, (x,))
        # Detach outputs to prevent gradients flowing back
        truth: torch.Tensor = outputs.detach()
        # Compute the loss without reduction
        return self.loss(reduction='none')(outputs, truth)

    def grad_along_u(
        self,
        x: torch.Tensor,
        w0: Dict[str, torch.Tensor],
        u: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # Compute the Jacobian of the loss with respect to parameters
        grad_f: Dict[str, torch.Tensor] = jacrev(self.compute_loss, argnums=1)(x, w0)
        # Compute the sum over the einsum operations for each parameter
        return sum(
            [
                einops.einsum(
                    grad_f[name],
                    u[name.replace('.', '__')],
                    'batch i ..., k ... -> batch k',
                )
                for name in grad_f.keys()
            ]
        )

    def double_grad_along_u(
        self, x: torch.Tensor, u: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Compute the second derivative (Hessian) along u
        jac: Dict[str, torch.Tensor] = jacrev(self.grad_along_u, argnums=1)(
            x, self.w0, u
        )
        # Compute the dot product of the Jacobian and u vectors
        return sum(
            [
                einops.einsum(
                    jac[name].flatten(2, -1),
                    u[name.replace('.', '__')].flatten(1, -1),
                    'batch k u, k u -> batch k',
                )
                for name in jac.keys()
            ]
        )

    def normalize_parameters(self) -> None:
        # Concatenate all parameters into a single tensor
        u_tensor: torch.Tensor = self.params_to_vectors(self._parameters)
        norms: torch.Tensor = u_tensor.norm(dim=1, keepdim=True)  # Norms per batch

        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'model' not in name:
                    # Reshape norms to match the dimensions of param (excluding batch dimension)
                    param_shape = param.shape[1:]
                    norms_reshaped: torch.Tensor = norms.view(
                        -1, *([1] * len(param_shape))
                    )
                    # Normalize parameters in-place
                    param.div_(norms_reshaped)

    def params_to_vectors(
        self, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Flatten and concatenate all parameters into a single tensor
        return torch.cat(
            [param.view(param.size(0), -1) for param in params.values()], dim=1
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Optionally normalize parameters
        # self.normalize_parameters()

        # Convert u parameters to a single tensor
        u_tensor: torch.Tensor = self.params_to_vectors(self._parameters)

        # Compute the double gradient along u
        dH_du: torch.Tensor = self.double_grad_along_u(x, self._parameters)

        return dH_du, u_tensor
        