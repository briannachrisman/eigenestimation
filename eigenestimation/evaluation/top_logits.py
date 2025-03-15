import torch
import einops
from torch.func import functional_call, jacfwd
import gc

def compute_jacobian(eigenmodel, sample, feature_idx, device='cuda', has_token_dim=True):
    """
    Computes the Jacobian of the reconstructed network with respect to `coef`.

    Parameters:
    - eigenmodel: The model with low-rank decomposition.
    - sample: The input sample tensor (should already be on `device`).
    - feature_idx: Index of the feature used in reconstruction.
    - device: 'cuda' or 'cpu' (default: 'cuda').

    Returns:
    - Jacobian tensor.
    """
    
    # Clear memory to avoid CUDA issues
    gc.collect()
    torch.cuda.empty_cache()

    def wrapper_fn(eigenmodel, coef, sample, feature_idx):
        """
        Efficiently reconstruct network weights using eigenmodel and functional_call.
        """
        coefficients = torch.zeros(eigenmodel.n_features, device=device)
        coefficients[feature_idx] = 1
        
        reconstruction = eigenmodel.add_to_network(coef * coefficients)
        # Use functional_call for inference with the reconstructed network
        return functional_call(eigenmodel.model, reconstruction, sample.unsqueeze(0)).softmax(dim=-1)

    # Ensure `coef` is on the correct device and requires gradient
    coef = torch.tensor(0.0, requires_grad=True, device=device)
    # Compute softmax output

    # Compute Jacobian efficiently using forward-mode differentiation
    if has_token_dim:
        jacobian_fn = jacfwd(lambda c: wrapper_fn(eigenmodel, c, sample, feature_idx)[0, -1, ...], randomness='same')
    else:
        jacobian_fn = jacfwd(lambda c: wrapper_fn(eigenmodel, c, sample, feature_idx)[0, ...], randomness='same')

    # Compute Jacobian
    jacobian = jacobian_fn(coef)

    return jacobian




