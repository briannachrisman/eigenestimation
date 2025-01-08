# Standard library imports
import importlib
import gc
import copy

# Third-party imports
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import einops
import matplotlib.pyplot as plt
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate


# Local imports
import toy_models.train
import toy_models.polytope

# Reload modules for interactive sessions

# Specific imports from local modules
from toy_models.polytope import ReluNetwork, GeneratePolytopeData
from toy_models.train import TrainModel

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'


n_inputs = 3
n_input_choices = 3
n_polytopes =  n_input_choices**n_inputs

n_hidden_layers = 5
n_hidden_neurons_per_layer = 5
n_samples = 10000
n_epochs = 10000

learning_rate = .001
print("Generating data...")
train_X, train_y, lookup_dict = GeneratePolytopeData(n_inputs, n_input_choices, n_samples)
model_dataloader = DataLoader(
    TensorDataset(train_X, train_y), batch_size=64, shuffle=True)
relu_network = ReluNetwork(n_inputs, n_polytopes, n_hidden_layers,n_hidden_neurons_per_layer).to(device)

print("Training...")
_, _, _ = TrainModel(
    relu_network, nn.NLLLoss(), learning_rate, model_dataloader, n_epochs=n_epochs, device=device)


# Save tms model to /models folder
torch.save(relu_network.state_dict(), 'models/polytope.pth')