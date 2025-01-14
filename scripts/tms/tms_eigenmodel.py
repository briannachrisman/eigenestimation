# Remember to login to wandb!
import sys
import os 

# Add the test directory to sys.path



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

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to the Python path
sys.path.insert(0, parent_dir)
print(sys.path)

from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate
import wandb


from eigenestimation.eigenhora import EigenHora
from eigenestimation import loss
from eigenestimation.train import Train
from evaluation.examples import TopActivatingSamples 
from evaluation.networks import DrawNeuralNetwork

from toy_models import tms
from eigenestimation.utils import TransformDataLoader, DeleteParams, RetrieveWandBArtifact

device = 'cuda'

# Load TMS model
tms_model = tms.AutoencoderSymmetric(input_dim=5, hidden_dim=2)
tms_model.load_state_dict(torch.load(f"{parent_dir}/models/tms_model.pth", weights_only=True))


# Load TMS model
tms_model_p = tms.AutoencoderParallel(input_dim=5, hidden_dim=2, n_networks=3)
tms_model_p.load_state_dict(torch.load(f"{parent_dir}/models/tms_model_parallel.pth", weights_only=True))

def model0(y):
    return torch.zeros_like(y)

n_networks = 3

n_features = 5

n_datapoints = 3*4096
X_tms_p, _, _= tms.GenerateTMSDataParallel(
    num_features=n_features, num_datapoints=n_datapoints,
    sparsity=.05,
    batch_size=16,
    n_networks=n_networks)

X_tms_p = X_tms_p.to(device)

hora_features = 15
hora_rank = 1
eigenmodel = EigenHora(tms_model_p.to(device), model0, loss.MSELoss(), hora_features, hora_rank, device=device).to(device)


dataloader = TransformDataLoader(X_tms_p, batch_size=32, transform_fn=eigenmodel.compute_jacobian)


eval_dataloader = TransformDataLoader(X_tms_p[:1000], batch_size=32, transform_fn=eigenmodel.compute_jacobian)
Train(eigenmodel, dataloader, lr=.001, n_epochs=2, L0_penalty=.01, device=device, project_name='eigenestimation', run_name='tms_model_parallel',
      eval_fns={TopActivatingSamples:[3]}, eval_dataloader=eval_dataloader)    

# Save model
torch.save(tms_model_p.state_dict(), 'models/tms_eigenmodeljob.pth')
