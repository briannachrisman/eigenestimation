import einops
import torch
from torch.utils.data import DataLoader
from typing import Tuple, List
import gc
import numpy 
import matplotlib.pyplot as plt
from torch.func import jvp
from functools import partial 

def PrintFeatureVals(X: torch.Tensor, eigenmodel: torch.nn.Module, device='cuda') -> None:
    # Compute dH_du and u_tensor from the model
    _, dH_du = eigenmodel(X.to(device), eigenmodel.u.to(device))    
    
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
    _, dH_du = eigenmodel(X.to(device), eigenmodel.u.to(device))
    
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
    dH_list = []
    bottom_logits_list = []
    top_logits_list = []

    dataloader_X = DataLoader(X, shuffle=False, batch_size=batch_size)
    with torch.no_grad():
        for X_batch in dataloader_X:

            f = partial(eigenmodel.compute_loss, X_batch)

            # Compute dH for the current minibatch
            _, H = jvp(f, 
                primals=(eigenmodel.param_dict,), 
                tangents=(eigenmodel.vector_to_parameters(eigenmodel.u[feature_idx]),))            #FIM_diag =  einops.einsum(dP_batch, dP_batch, '... c, ... c -> ...')
            
            dH_list.append(H)

            #dP_batch = eigenmodel_transformer(X_transformer[:4].to(device), eigenmodel_transformer.u[[1]])[0].detach()
            #top_idx = torch.topk(dP_batch, k_logits, dim=0, largest=True, sorted=True).indices
            #bottom_idx = torch.topk(dP_batch, k_logits, dim=0, largest=False, sorted=True).indices
            #top_logits_list.append(top_idx)
            #bottom_logits_list.append(bottom_idx)

        
            #torch.cuda.empty_cache()
            #gc.collect()
    
    if True:#for f in feature_idx:

        # Concatenate dH results from all minibatches
        feature_vals = (torch.cat(dH_list, dim=0))
        #top_logits_idx = torch.cat(top_logits_list, dim=1)
        #bottom_logits_idx = torch.cat(bottom_logits_list, dim=1)

        # Flatten the tensor to find the top k values globally
        flattened_tensor = feature_vals.flatten()
        #print(top_logits_idx.shape)
        # Find the top 5 highest values and their indices in the flattened tensor
        top_values, top_idx = torch.topk(flattened_tensor, top_k, largest=False)
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
            bolded_tokens = eigenmodel.model.tokenizer.convert_tokens_to_string(tokens_list[:(token+1)])
            bolded_tokens = bolded_tokens.replace("\n", "newline")

            # Decode the specific token with the highest value
            token_of_value = eigenmodel.model.tokenizer.decode(X[sample, token]).replace("\n", "newline")
            
            #top_logits =  [eigenmodel.model.tokenizer.decode(i) for i in (top_logits_idx[:,sample,token])]
            #bottom_logits =  [eigenmodel.model.tokenizer.decode(i) for i in (bottom_logits_idx[:,sample,token])]


            # Print the modified tokens with the bolded token and its value
            print(f"{bolded_tokens} -> {token_of_value} (Value: {value:.3f})")# top: {top_logits}, bottom: {bottom_logits}")




import matplotlib.pyplot as plt
import numpy as np

def DrawNeuralNetwork(weights_dict):
    """
    Draw a neural network diagram based on a dictionary of weights.
    
    Args:
        weights_dict (dict): Dictionary where keys are layer names and values are weight matrices (tensors).
                             Each weight matrix should have dimensions (output_size, input_size).
    """
    # Get layer names and sizes based on the weight matrices
    layer_names = list(weights_dict.keys())
    input_size = weights_dict[layer_names[0]].shape[0]
    layer_sizes = [input_size] + [weights_dict[layer].shape[1] for layer in layer_names]
    
    # Define x-coordinates for each layer
    layer_x = np.linspace(1, len(layer_sizes), len(layer_sizes))
    
    # Define y-coordinates for each layer's nodes, spacing them vertically
    layer_y = {f'layer_{i}': np.linspace(0.1, 0.9, layer_sizes[i]) for i in range(len(layer_sizes))}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axis('off')  # Turn off the axis
    
    # Draw the nodes for each layer
    def draw_layer_nodes(layer_x, layer_y, label, text_x):
        for y in layer_y:
            ax.plot(layer_x, y, 'o', markersize=12, color='skyblue')
        ax.text(text_x, 1.0, label, ha='center', fontsize=12, color='black')

    # Draw connections (edges) between nodes based on weights
    def draw_connections(layer_x1, layer_y1, layer_x2, layer_y2, weights):
        for i, y1 in enumerate(layer_y1):
            for j, y2 in enumerate(layer_y2):
                weight = weights[i, j]
                color = 'green' if weight > 0 else 'red'
                linewidth =  1*abs(weight)  # Scale line width by weight magnitude
                ax.plot([layer_x1, layer_x2], [y1, y2], color=color, linewidth=linewidth)
    
    # Draw layers and connections iteratively
    for i, (layer_name, weights) in enumerate(weights_dict.items()):
        # Draw the nodes for the current layer
        if i == 0:
            draw_layer_nodes(layer_x[0], layer_y[f'layer_{i}'], 'Input', layer_x[0])
        draw_layer_nodes(layer_x[i + 1], layer_y[f'layer_{i + 1}'], layer_name, (layer_x[i + 1] + layer_x[i])/2)
        
        # Draw connections from the previous layer to the current layer
        draw_connections(layer_x[i], layer_y[f'layer_{i}'], layer_x[i + 1], layer_y[f'layer_{i + 1}'], weights.cpu().detach().numpy())
    
    plt.show()