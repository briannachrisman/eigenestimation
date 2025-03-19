import argparse
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
import wandb  # Add Weights & Biases for tracking
import os
import torchvision
from functorch.experimental import replace_all_batch_norm_modules_
from functools import partial
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import random
import json


param_dict = {
    'features.7.block.0.0.weight': [12, 4, 1, 1],
    'features.7.block.0.1.weight': [12],
    'features.7.block.0.1.bias': [12],
    'features.7.block.1.0.weight': [12, 1, 5, 5],
    'features.7.block.1.1.weight': [12],
    'features.7.block.1.1.bias': [12],
    'features.7.block.2.fc1.weight': [3, 12, 1, 1],
    'features.7.block.2.fc1.bias': [3],
    'features.7.block.2.fc2.weight': [12, 3, 1, 1],
    'features.7.block.2.fc2.bias': [12],
    'features.7.block.3.0.weight': [4, 12, 1, 1],
    'features.7.block.3.1.weight': [4],
    'features.7.block.3.1.bias': [4]
}

import sys
import transformer_lens
from transformers import AutoTokenizer
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset

# Append module directory for imports
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eigenestimation"))
sys.path.append(module_dir)

from eigenestimation.eigenmodel.trainer import Trainer
from eigenestimation.eigenmodel.eigenmodel import EigenModel
from eigenestimation.utils.utils import TransformDataLoader
from eigenestimation.utils.loss import MSELoss, MSEVectorLoss, KLDivergenceSumOverTokensLoss, KLDivergenceNoFlattenLoss

# Ensure correct device usage
device = "cuda" if torch.cuda.is_available() else "cpu"

from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")
from eigenestimation.utils.uniform_models import ZeroOutput, UniformProbs

from eigenestimation.eigenmodel.trainer import Trainer
from eigenestimation.eigenmodel.eigenmodel import EigenModel
from eigenestimation.utils.utils import TransformDataLoader, DeleteParams, RetrieveWandBArtifact


from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


from eigenestimation.toy_models.cnn_wrapper import CNNWrapper, ImageOnlyDataset
# Ensure correct device usage
device = "cuda" if torch.cuda.is_available() else "cpu"

from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")
from eigenestimation.utils.uniform_models import UniformLogits

torch.manual_seed(42)
def get_args_parser():
    """
    Parses command-line arguments for configuring the training process.

    Returns:
        argparse.ArgumentParser: Parsed arguments object.
    """
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--lr-step-epochs", type=int, default=100, help="Learning rate step epochs")
    parser.add_argument("--lr-decay-rate", type=float, default=0.8, help="Learning rate step factor")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--token-length", type=int, default=8, help="Batch size for training")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Directory to save checkpoints")
    parser.add_argument("--warm-start-epochs", type=int, default=0, help="Number of warm start epochs")
    
    parser.add_argument("--dataset", type=str, default='roneneldan/TinyStories', help="Dataset to use")
    parser.add_argument("--train-split", type=str, default='train[:1%]', help="Train split to use")
    parser.add_argument("--eval-split", type=str, default='validation[:1%]', help="Eval split to use")
    
    parser.add_argument("--n-train-samples", type=int, default=100, help="Number of train samples to use")
    parser.add_argument("--n-eval-samples", type=int, default=100, help="Number of eval samples to use")

    
    parser.add_argument("--checkpoint-epochs", type=int, required=True, help="Frequency at which to save checkpoints")
    
    parser.add_argument("--log-epochs", type=int, required=True, help="Frequency at which to save checkpoints")
    
    parser.add_argument("--n-features", type=int, default=5, help="Number of input features")
    parser.add_argument("--n-hidden", type=int, default=2, help="Number of hidden features")
    
    parser.add_argument("--model", type=str, default='roneneldan/TinyStories-1M', help="Model to use")
    parser.add_argument("--tokenizer", type=str, default='EleutherAI/gpt-neo-125M', help="Tokenizer to use")
    
    parser.add_argument("--n-eigenfeatures", type=int, default=2, help="Number of networks")
    
    parser.add_argument("--top-k", type=float, default=.1, help="Top k percent of jvp values to keep")
    parser.add_argument("--chunk-size", type=int, default=10, help="Chunk size for computing gradients")
    
    parser.add_argument("--sparsity", type=float, default=0.1, help="Sparsity level for generated data")
    
    parser.add_argument("--wandb-project", type=str, default="tms-autoencoder-training", help="Weights & Biases project name")
    return parser

def setup_distributed_training(args):
    """
    Sets up the distributed training environment.

    Args:
        args (argparse.Namespace): Training configuration arguments.

    Returns:
        int: Global rank of the current process.
    """
    args.device_id = int(os.environ["LOCAL_RANK"])  # Assign GPU to this process
    torch.cuda.set_device(args.device_id)  # Assign GPU to this process
    dist.init_process_group("nccl")  # Expects environment variables to be set
    rank = int(os.environ["RANK"])
    args.is_master = rank == 0  # Check if the current process is the master
    return rank

def main(args, timer):
    """
    Main function to initialize data, model, and trainer, and begin training.
    """
    rank = setup_distributed_training(args)
    
    # Initialize Weights & Biases (only in the master process)
    if args.is_master:
        wandb.init(project=args.wandb_project, config=vars(args))

    # Hyperparameters and model configuration
    batch_size = args.batch_size

    model = torchvision.models.mobilenet_v3_small(pretrained=True).to(args.device_id)
    model.eval()

    # read in args.param_dict as dictionary
    vals_to_keep = param_dict.keys()
    
    # Make the eigenestimation a little smaller but only looking at a subset of the parameters.
    # Pick a random subset of tensors to include in paramters, and turn the rest into frozen buffers.
    params_to_delete = [name for name, param in model.named_parameters()]
    params_to_delete = [p for p in params_to_delete if p not in vals_to_keep]

    DeleteParams(model, params_to_delete)
    
    for n,p in model.named_parameters(): print(n, p.shape, p.numel())
    print(sum([p.numel() for p in model.parameters()]), 'parameters')

    # Define transformations for normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((224, 224), antialias=True)  # Normalize to [-1, 1]
    ])

    # Load CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR100(root="./data", train=True, transform=transform, download=True,)

    # Randomly select 100 indices
    random_indices = torch.randperm(len(dataset))[:args.n_train_samples]



    X_train = torch.stack([i[0] for i in Subset(dataset, random_indices)], dim=0)
   # Randomly select 100 indices
    
    random_indices = random.sample(range(len(dataset)), args.n_eval_samples)

    X_eval = ImageOnlyDataset(Subset(dataset, random_indices), device=args.device_id)
    
    del dataset

        
    print('device', device)
    eigenmodel = EigenModel(model, UniformLogits, KLDivergenceNoFlattenLoss(), 
                            args.n_eigenfeatures, param_dict)
    
    
    # Initialize the trainer and start training
    trainer = Trainer(eigenmodel, X_train, X_eval, args, timer)
    trainer.train()

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
    if args.is_master:
        timer.report("Finished!")
        wandb.finish()
