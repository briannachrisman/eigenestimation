import torch
import einops
from torch.func import functional_call, jacfwd
import gc

def compute_jacobian(eigenmodel, sample, feature_idx, token_idx, device='cuda'):
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

    def reconstruct_network(eigenmodel, coef, sample, feature_idx, token_idx):
        """
        Efficiently reconstruct network weights using eigenmodel and functional_call.
        """
        # Direct access to low-rank decomposed weights
        low_rank_decode = eigenmodel.low_rank_decode
        reconstruction = {}
        
        for name in low_rank_decode:
            # Extract the first component
            recon = low_rank_decode[name][0][..., feature_idx]

            # Efficient einsum application
            for tensor in low_rank_decode[name][1:]:
                recon = einops.einsum(recon, tensor[..., feature_idx], '... r, w r -> ... w r')

            # Final summation
            reconstruction[name] = einops.einsum(recon, '... w r -> ... w') * coef

        # Use functional_call for inference with the reconstructed network
        return functional_call(eigenmodel.model, reconstruction, sample)

    # Ensure `coef` is on the correct device and requires gradient
    coef = torch.tensor(0.0, requires_grad=True, device=device)

    # Compute softmax output

    # Compute Jacobian efficiently using forward-mode differentiation
    jacobian_fn = jacfwd(lambda c: reconstruct_network(eigenmodel, c, sample, feature_idx)[0, token_idx, :].softmax(dim=-1))

    # Compute Jacobian
    jacobian = jacobian_fn(coef)

    return jacobian

