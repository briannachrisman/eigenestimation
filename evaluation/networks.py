import matplotlib.pyplot as plt
import numpy as np
import einops
import torch
from torch.utils.data import DataLoader
from typing import Tuple, List
import gc
import numpy 
import matplotlib.pyplot as plt
from torch.func import jvp
from functools import partial 

def DrawNeuralNetwork(weights_dict, biases_dict, title=''):
    """
    Draw a neural network diagram based on a dictionary of weights and biases.

    Args:
        weights_dict (dict): Dictionary where keys are layer names and values are weight matrices (tensors).
                             Each weight matrix should have dimensions (output_size, input_size).
        biases_dict (dict): Dictionary where keys are layer names and values are bias vectors (tensors).
                            Each bias vector should have dimensions (output_size,).
    """
    # Get layer names and sizes based on the weight matrices
    layer_names = list(weights_dict.keys())
    input_size = weights_dict[layer_names[0]].shape[1]  # Input size from the first layer's weights
    layer_sizes = [input_size] + [weights_dict[layer].shape[0] for layer in layer_names]  # Output sizes define nodes per layer

    # Define x-coordinates for each layer
    layer_x = np.linspace(1, len(layer_sizes), len(layer_sizes))

    # Define y-coordinates for each layer's nodes, spacing them vertically
    layer_y = {f'layer_{i}': np.linspace(0.1, 0.9, layer_sizes[i]) for i in range(len(layer_sizes))}

    # Create figure
    max_y = .2*max(layer_sizes)
    fig, ax = plt.subplots(figsize=(len(weights_dict), max_y))
    ax.axis('off')  # Turn off the axis

    # Draw the nodes for each layer
    def draw_layer_nodes(layer_x, layer_y, label, text_x, biases=None):
        for idx, y in enumerate(layer_y):
            # Normalize bias darkness
            bias_value = biases[idx] if biases is not None else 0
            color = (1, 0, 0, abs(bias_value)) if bias_value > 0 else (0, 0, 1, abs(bias_value))
            
            ax.plot(layer_x, y, 'o', markersize=8, color=color, markeredgecolor='black', markeredgewidth=.5)
        ax.text(text_x, 1.0, label, ha='center', fontsize=5, color='black')

    # Draw connections (edges) between nodes based on weights
    def draw_connections(layer_x1, layer_y1, layer_x2, layer_y2, weights):
        for i, y1 in enumerate(layer_y1):
            for j, y2 in enumerate(layer_y2):
                weight = weights[j, i]  # Note the order (output, input)
                color = 'red' if weight > 0 else 'blue'
                linewidth = 1 * abs(weight)  # Scale line width by weight magnitude
                ax.plot([layer_x1, layer_x2], [y1, y2], color=color, linewidth=linewidth)

    # Draw layers and connections iteratively
    for i, (layer_name, weights) in enumerate(weights_dict.items()):
        biases = biases_dict.get(layer_name, torch.zeros(weights.shape[0])).cpu().detach().numpy()

        # Draw the nodes for the current layer
        if i == 0:
            draw_layer_nodes(layer_x[0], layer_y[f'layer_{i}'], '', layer_x[0])
        draw_layer_nodes(layer_x[i + 1], layer_y[f'layer_{i + 1}'], layer_name, (layer_x[i + 1] + layer_x[i]) / 2, biases=biases)

        # Draw connections from the previous layer to the current layer
        draw_connections(layer_x[i], layer_y[f'layer_{i}'], layer_x[i + 1], layer_y[f'layer_{i + 1}'], weights.cpu().detach().numpy())

    plt.title(title, y=1.1, fontsize=5)
    plt.show()
    return fig
    