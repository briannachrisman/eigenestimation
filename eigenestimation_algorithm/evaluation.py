import torch
from typing import Tuple

def PrintFeatureVals(X: torch.Tensor, eigenmodel: torch.nn.Module) -> None:
    # Compute dH_du and u_tensor from the model
    dH_du, _ = eigenmodel(X)
    
    # Print rounded values of the input features and corresponding outputs
    for x, h in zip(X.detach().cpu().numpy().round(2), dH_du.detach().cpu().numpy().round(2)):
        print(x, '-->', h)

def ActivatingExamples(
    X: torch.Tensor,
    eigenmodel: torch.nn.Module,
    idx: int,
    k: int,
    ascending: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute dH_du from the model
    dH_du, _ = eigenmodel(X)
    
    # Select the specific index and convert to numpy
    dH_du_idx: torch.Tensor = dH_du[:, idx].detach().cpu().numpy()
    
    # Sort indices based on values (ascending or descending)
    argidx = dH_du_idx.argsort()
    if not ascending:
        argidx = argidx[::-1]

    # Return the top k examples and their corresponding values
    top_k_examples = X.detach().cpu().numpy()[argidx[:k], :]
    top_k_values = dH_du_idx[argidx[:k]]

    return top_k_examples, top_k_values
