import einops
import torch
from torch.utils.data import DataLoader
from typing import Tuple, List
import gc
def PrintFeatureVals(X: torch.Tensor, eigenmodel: torch.nn.Module, device='cuda') -> None:
    # Compute dH_du and u_tensor from the model
    dH_du = eigenmodel(X.to(device), eigenmodel.u.to(device)).detach()
    
    # Print rounded values of the input features and corresponding outputs
    for x, h in zip(X.detach().cpu().numpy().round(2), dH_du.transpose(0,1).detach().cpu().numpy().round(2)):
        print(x, '-->\n', h)

def ActivatingExamples(
    X: torch.Tensor,
    eigenmodel: torch.nn.Module,
    idx: int,
    k: int,
    ascending: bool = False, device='cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute dH_du from the model
    dH_du = eigenmodel(X.to(device), eigenmodel.u.to(device)).detach()
    
    # Select the specific index and convert to numpy
    dH_du_idx: torch.Tensor = dH_du[idx,:].detach().cpu().numpy().flatten()
    
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
    batch_size: int = 32,  # Define a batch size for minibatch processing,
    device: str = 'cuda', 
    k_logits: int = 5
) -> None:
    num_samples = X.shape[0]
    
    # Split X into minibatches
    dH_list: List[torch.Tensor] = []
    bottom_logits_list = []
    top_logits_list = []

    dataloader_X = DataLoader(X, shuffle=False, batch_size=batch_size)
    with torch.no_grad():
        for X_batch in dataloader_X:
            
            # Compute dH for the current minibatch
            dP_batch = eigenmodel(X_batch.to(device), eigenmodel.u[[feature_idx]])[0].detach()
            FIM_diag =  einops.einsum(dP_batch, dP_batch, '... c, ... c -> ...')
            dH_list.append(FIM_diag)


            #dP_batch = eigenmodel_transformer(X_transformer[:4].to(device), eigenmodel_transformer.u[[1]])[0].detach()
            top_idx = torch.topk(dP_batch, k_logits, dim=-1, largest=True, sorted=True).indices
            bottom_idx = torch.topk(dP_batch, k_logits, dim=-1, largest=False, sorted=True).indices
            top_logits_list.append(top_idx)
            bottom_logits_list.append(bottom_idx)

        
            #torch.cuda.empty_cache()
            #gc.collect()
        
    # Concatenate dH results from all minibatches
    feature_vals = torch.cat(dH_list, dim=0)
    top_logits_idx = torch.cat(top_logits_list, dim=0)
    bottom_logits_idx = torch.cat(bottom_logits_list, dim=0)

    # Flatten the tensor to find the top k values globally
    flattened_tensor = feature_vals.flatten()
    
    # Find the top 5 highest values and their indices in the flattened tensor
    top_values, top_idx = torch.topk(flattened_tensor, top_k)
    # Convert the flattened indices back to the original 3D indices
    top_idx_sample, top_idx_token = torch.unravel_index(top_idx, feature_vals.shape)


    # Iterate over the top values and their indices
    for (sample, token, value) in zip(top_idx_sample, top_idx_token, top_values):
        #print(sample, token, value)
        # Decode the entire sequence of tokens for the current sample as individual tokens
        tokens_list = eigenmodel.model.tokenizer.convert_ids_to_tokens(X[sample].tolist())
        
        # Bold the token at the specific index
        tokens_list[token] = f"**{tokens_list[token]}**"
        
        # Join the tokens back together for displaying
        bolded_tokens = eigenmodel.model.tokenizer.convert_tokens_to_string(tokens_list)
        bolded_tokens = bolded_tokens.replace("\n", "newline")

        # Decode the specific token with the highest value
        token_of_value = eigenmodel.model.tokenizer.decode(X[sample, token]).replace("\n", "newline")
        
        top_logits =  [eigenmodel.model.tokenizer.decode(i) for i in (top_logits_idx[sample,token,:])]
        bottom_logits =  [eigenmodel.model.tokenizer.decode(i) for i in (bottom_logits_idx[sample,token,:])]


        # Print the modified tokens with the bolded token and its value
        print(f"{bolded_tokens} -> {token_of_value} (Value: {value:.3f}), top: {top_logits}, bottom: {bottom_logits}")

