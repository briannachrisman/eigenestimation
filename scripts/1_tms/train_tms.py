import argparse
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
import wandb  # Add Weights & Biases for tracking
import os
import sys
import numpy as np
from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")

from eigenestimation.toy_models.trainer import Trainer
from eigenestimation.toy_models.tms import AutoencoderSymmetric  # Import your model
from eigenestimation.toy_models.data import GenerateTMSInputs

# Ensure correct device usage
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
np.random.seed(42)
def get_args_parser():
    """
    Parses command-line arguments for configuring the training process.

    Returns:
        argparse.ArgumentParser: Parsed arguments object.
    """
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--lr-step-epochs", type=int, default=10, help="Learning rate step epochs")
    parser.add_argument("--lr-decay-rate", type=float, default=1, help="Learning rate step size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-epochs", type=int, required=True, help="Frequency at which to save checkpoints")
    
    parser.add_argument("--log-epochs", type=int, required=True, help="Frequency at which to save checkpoints")
    
    parser.add_argument("--n-features", type=int, default=5, help="Number of input features")
    parser.add_argument("--n-hidden", type=int, default=2, help="Number of hidden features")
    parser.add_argument("--n-training-datapoints", type=int, default=100, help="Number of training data points")
    parser.add_argument("--n-eval-datapoints", type=int, default=100, help="Number of evaluation data points")
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
    n_features = args.n_features
    n_hidden = args.n_hidden
    n_training_datapoints = args.n_training_datapoints
    n_eval_datapoints = args.n_eval_datapoints
    sparsity = args.sparsity
    batch_size = args.batch_size

    # Generate training and evaluation data
    
    # If X_train.shape[0] < batch_size, sample again
    X_train  = GenerateTMSInputs(num_features=n_features, num_datapoints=n_training_datapoints, sparsity=sparsity)
    y_train = X_train
    
    X_eval  = GenerateTMSInputs(num_features=n_features, num_datapoints=n_eval_datapoints, sparsity=sparsity)
    y_eval = X_eval 
    
    
    # Create TensorDatasets for DataLoader compatibility
    train_dataset = TensorDataset(X_train, y_train)
    eval_dataset = TensorDataset(X_eval, y_eval)

    # Initialize the model and loss function
    model = AutoencoderSymmetric(n_features, n_hidden).to(device)
    criterion = nn.MSELoss()

    # Initialize the trainer and start training
    trainer = Trainer(model, criterion, train_dataset, eval_dataset, args, timer)
    trainer.train()
        

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
    if args.is_master:
        timer.report("Finished!")
        wandb.finish()
    
