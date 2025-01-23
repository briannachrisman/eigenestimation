# eigenestimation.py
import torch
import torch.nn as nn
from typing import Callable, List
from torch import Tensor

class CustomMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int) -> None:
        """
        Custom MLP with ReLU activations for each subnetwork.

        Args:
            input_dim (int): Input dimension of the MLP.
            hidden_dims (List[int]): List of hidden layer dimensions.
            output_dim (int): Output dimension of the MLP.
        """
        super(CustomMLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.Linear(dims[-1], output_dim))
        
        # Initialize weights
        for layer in layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # Register layer parametrs with names
        for i, layer in enumerate(layers):
            self.add_module(f"{i}", layer)
        
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        return self.layers[-1](x)
    
    



class ParallelSerializedModel(nn.Module):
    def __init__(self, 
                 parallel_layers: List[List[nn.Module]], 
                 device: str = 'cuda') -> None:
        """
        A model with multiple parallel and serialized layers.

        Args:
            parallel_layers (List[List[nn.Module]]): A list of lists, where each inner list contains 
                parallel layers for a single stage of processing.
            n_inputs (int): Number of input features.
            device (str, optional): Device for computations. Defaults to 'cuda'.
        """
        super(ParallelSerializedModel, self).__init__()

        self.parallel_layers = nn.ModuleList([nn.ModuleList(stage) for stage in parallel_layers])
        self.device = device
        
        # Initialize all paramters to be uniform between -1 and 1
        for layer_i, layer in enumerate(self.parallel_layers):
            for sublayer_i, sublayer in enumerate(layer):
                for linear_layer_i, linear_layer in enumerate(sublayer.layers):
                    if isinstance(linear_layer, nn.Linear):
                        nn.init.uniform_(linear_layer.weight, 0,1)#, -1 ,1)
                        nn.init.zeros_(linear_layer.bias)
                        
                        self.register_parameter(f"layer_{layer_i}_sublayer_{sublayer_i}_linear_{linear_layer_i}_weight", linear_layer.weight)
                        self.register_parameter(f"layer_{layer_i}_sublayer_{sublayer_i}_linear_{linear_layer_i}_bias", linear_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, n_inputs).

        Returns:
            Tensor: Output tensor of shape (batch_size, n_outputs).
        """
        subnetwork_ouputs = []
        for stage in self.parallel_layers:
            # Collect outputs from all parallel layers in the current stage
            stage_outputs = []
            layer_output = []
            for layer in stage:
                out = layer(x)
                stage_outputs.append(out)
                layer_output.append(out)
            subnetwork_ouputs.append(layer_output)

            # Combine the outputs (e.g., concatenate or sum)
            x = torch.cat(stage_outputs, dim=-1)

        return x


    def subnetwork_outputs(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, n_inputs).

        Returns:
            Tensor: Output tensor of shape (batch_size, n_outputs).
        """
        subnetwork_ouputs = []
        for stage in self.parallel_layers:
            # Collect outputs from all parallel layers in the current stage
            stage_outputs = []
            layer_output = []
            for layer in stage:
                out = layer(x)
                stage_outputs.append(out)
                layer_output.append(out)
            subnetwork_ouputs.append(layer_output)

            # Combine the outputs (e.g., concatenate or sum)
            x = torch.cat(stage_outputs, dim=-1)

        return subnetwork_ouputs