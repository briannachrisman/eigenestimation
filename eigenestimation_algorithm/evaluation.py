import einops
import torch
from typing import Tuple, List

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




import torch
import einops
from typing import List

def PrintFeatureValsTransformer(
    eigenmodel: torch.nn.Module,
    X: torch.Tensor,
    feature_idx: int,
    batch_size: int = 10  # Define a batch size for minibatch processing
) -> None:
    device = X.device  # Assume the tensor is already on the appropriate device
    num_samples = X.shape[0]
    
    # Split X into minibatches
    dH_list: List[torch.Tensor] = []
    
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        X_batch = X[start:end]

        # Compute dH for the current minibatch
        dH_batch, _ = eigenmodel(X_batch)
        dH_list.append(dH_batch)
    
    # Concatenate dH results from all minibatches
    dH = torch.cat(dH_list, dim=0)

    # Rearrange dH for feature extraction
    feature_vals = einops.rearrange(dH, '(s t) f -> s t f', s=X.shape[0], t=X.shape[1])[:, :, feature_idx]
    
    # Print feature values for each input in X
    for x, f in zip(X, feature_vals):
        f = f.detach().cpu().numpy().round(3)
        tokens = eigenmodel.model.tokenizer.decode(x)
        print(tokens, '->', f)

def PrintActivatingExamplesTransformer(
    eigenmodel: torch.nn.Module,
    X: torch.Tensor,
    feature_idx: int,
    top_k: int,
    batch_size: int = 10  # Define a batch size for minibatch processing
) -> None:

    # Split X into minibatches
    dH_list: List[torch.Tensor] = []
    
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        X_batch = X[start:end]

        # Compute dH for the current minibatch
        dH_batch, _ = eigenmodel(X_batch)
        dH_list.append(dH_batch)
    
    # Concatenate dH results from all minibatches
    dH = torch.cat(dH_list, dim=0)

    feature_vals = einops.rearrange(dH, '(s t) f -> s t f', s=X.shape[0], t=X.shape[1])[:,:,feature_idx]
    # Flatten the tensor to find the top k values globally
    flattened_tensor = feature_vals.flatten()
    
    # Find the top 5 highest values and their indices in the flattened tensor
    top_values, top_idx = torch.topk(flattened_tensor, top_k)

    # Convert the flattened indices back to the original 3D indices
    top_idx_sample, top_idx_token = torch.unravel_index(top_idx, feature_vals.shape)


    # Iterate over the top values and their indices
    for (sample, token, value) in zip(top_idx_sample, top_idx_token, top_values):
        # Rearrange dH for feature extraction
        feature_vals = einops.rearrange(dH, '(s t) f -> s t f', s=X.shape[0], t=X.shape[1])[:, :, feature_idx]

        # Decode the entire sequence of tokens for the current sample as individual tokens
        tokens_list = eigenmodel.model.tokenizer.convert_ids_to_tokens(X[sample].tolist())
        
        # Bold the token at the specific index
        tokens_list[token] = f"**{tokens_list[token]}**"
        
        # Join the tokens back together for displaying
        bolded_tokens = eigenmodel.model.tokenizer.convert_tokens_to_string(tokens_list)

        # Decode the specific token with the highest value
        token_of_value = eigenmodel.model.tokenizer.decode(X[sample, token])
        
        # Print the modified tokens with the bolded token and its value
        print(f"{bolded_tokens} -> {token_of_value} (Value: {value:.3f})")


