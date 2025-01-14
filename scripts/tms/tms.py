

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
import toy_models.tms
import toy_models.train
import toy_models.transformer_wrapper
import eigenestimation_algorithm.train
import eigenestimation_algorithm.eigenestimation
import eigenestimation_algorithm.evaluation

# Reload modules for interactive sessions
importlib.reload(toy_models.xornet)
importlib.reload(toy_models.tms)
importlib.reload(toy_models.train)
importlib.reload(toy_models.transformer_wrapper)
importlib.reload(eigenestimation_algorithm.train)
importlib.reload(eigenestimation_algorithm.eigenestimation)
importlib.reload(eigenestimation_algorithm.evaluation)

# Specific imports from local modules
from toy_models.xornet import XORNet, GenerateXORData, XORNetParallel, GenerateXORDataParallel
from toy_models.tms import Autoencoder, AutoencoderSymmetric, GenerateTMSData, AutoencoderParallel, GenerateTMSDataParallel
from toy_models.train import TrainModel
from toy_models.transformer_wrapper import TransformerWrapper, DeleteParams, KLDivergenceLoss
from eigenestimation_algorithm.eigenestimation import EigenEstimation, EigenEstimationComparison
from eigenestimation_algorithm.train import TrainEigenEstimation, TrainEigenEstimationComparison
from eigenestimation_algorithm.evaluation import (
    PrintFeatureVals,
    ActivatingExamples,
    PrintFeatureValsTransformer,
    PrintActivatingExamplesTransformer,
    DrawNeuralNetwork
)

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'


import eigenestimation_algorithm.gradientextractor
importlib.reload(eigenestimation_algorithm.gradientextractor)

from eigenestimation_algorithm.gradientextractor import (
    JacobianExtractor,
    ExtractJacs,
) 



n_features = 5
hidden_dim = 2
n_datapoints = 5000
sparsity = .05

batch_size = 16
learning_rate = .01
n_epochs = 500
torch.manual_seed(42)

X_tms, Y_tms, dataloader_TMS = GenerateTMSData(
    num_features=n_features, num_datapoints=n_datapoints, sparsity=sparsity, batch_size=batch_size)
tms_model = AutoencoderSymmetric(n_features, hidden_dim).to(device)
_, _, _ = TrainModel(tms_model, nn.MSELoss(), learning_rate, dataloader_TMS, n_epochs=n_epochs, device=device)


# Save tms model to /models folder
torch.save(tms_model.state_dict(), 'models/tms_model.pth')




torch.manual_seed(42)
n_networks = 3
n_features = 5

n_datapoints = 3*4096
X_tms_p, Y_tms_p, dataloader_tms_p = GenerateTMSDataParallel(
    num_features=n_features, num_datapoints=n_datapoints, sparsity=sparsity, batch_size=batch_size, n_networks=n_networks)
tms_model_p = AutoencoderParallel(n_features, hidden_dim, n_networks).to(device)


for n,p in tms_model.named_parameters():
    if "W" in n:
        dict(tms_model_p.named_parameters())['W_in'].data = torch.block_diag(*[p for _ in range(n_networks)])
        dict(tms_model_p.named_parameters())['W_out'].data = torch.block_diag(*[p.transpose(0,1) for _ in range(n_networks)])

    if "b" in n:
        dict(tms_model_p.named_parameters())[n].data = torch.concat([p for _ in range(n_networks)])


params_to_delete = [name for name, param in tms_model_p.named_parameters() if "W" not in name]

DeleteParams(tms_model_p, params_to_delete)

print(dict(tms_model_p.named_parameters()))


n_networks = 3
n_features = 5

n_datapoints = 3*4096
X_tms_p, Y_tms_p, dataloader_tms_p = GenerateTMSDataParallel(
    num_features=n_features, num_datapoints=n_datapoints, sparsity=sparsity, batch_size=batch_size, n_networks=n_networks)
tms_model_p = AutoencoderParallel(n_features, hidden_dim, n_networks).to(device)


for n,p in tms_model.named_parameters():
    if "W" in n:
        dict(tms_model_p.named_parameters())['W_in'].data = torch.block_diag(*[p for _ in range(n_networks)])
        dict(tms_model_p.named_parameters())['W_out'].data = torch.block_diag(*[p.transpose(0,1) for _ in range(n_networks)])

    if "b" in n:
        dict(tms_model_p.named_parameters())[n].data = torch.concat([p for _ in range(n_networks)])


params_to_delete = [name for name, param in tms_model_p.named_parameters() if "W" not in name]

DeleteParams(tms_model_p, params_to_delete)

print(dict(tms_model_p.named_parameters()))

# Save model
torch.save(tms_model_p.state_dict(), 'models/tms_model_parallel.pth')
