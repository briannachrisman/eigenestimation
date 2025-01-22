import argparse
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
import wandb  # Add Weights & Biases for tracking
import os
import sys

from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")

# Append module directory for imports
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eigenestimation"))
sys.path.append(module_dir)

from toy_models.trainer import Trainer
from toy_models.polytope import ReluNetwork, GeneratePolytopeData  # Import your model

# Ensure correct device usage
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_args_parser():
    """
    Parses command-line arguments for configuring the training process.

    Returns:
        argparse.ArgumentParser: Parsed arguments object.
    """
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Directory to save checkpoints")
    parser.add_argument("--data-path", type=Path, required=True, help="Directory to save data")

    parser.add_argument("--checkpoint-epochs", type=int, required=True, help="Frequency at which to save checkpoints")
    
    parser.add_argument("--log-epochs", type=int, required=True, help="Frequency at which to save checkpoints")
    
    parser.add_argument("--n-features", type=int, default=2, help="Number of input features")
    parser.add_argument("--n-feature-choices", type=int, default=5, help="Number of input features")
    parser.add_argument("--n-hidden-units", type=int, default=2, help="Number of hidden units")
    parser.add_argument("--n-hidden-layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--n-training-datapoints", type=int, default=100, help="Number of training data points")
    parser.add_argument("--n-eval-datapoints", type=int, default=100, help="Number of evaluation data points")
    parser.add_argument("--wandb-project", type=str, default="polytope-training", help="Weights & Biases project name")
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

    # Generate training and evaluation data
    X, y, lookup_dict = GeneratePolytopeData(
        args.n_features,
        args.n_feature_choices, 
        (args.n_training_datapoints + args.n_eval_datapoints)
        )
    torch.save(lookup_dict, args.data_path)
    X_train, y_train = X[:args.n_training_datapoints], y[:args.n_training_datapoints]
    X_eval, y_eval = X[-args.n_eval_datapoints:], y[-args.n_eval_datapoints:]
    
    
    # Create TensorDatasets for DataLoader compatibility
    train_dataset = TensorDataset(X_train, y_train)
    eval_dataset = TensorDataset(X_eval, y_eval)

    # Initialize the model and loss function
    model = ReluNetwork(
        args.n_features,
        args.n_feature_choices**args.n_features,
        args.n_hidden_layers, args.n_hidden_units).to(device)
    
    criterion = nn.CrossEntropyLoss()

    # Initialize the trainer and start training
    trainer = Trainer(model, criterion, train_dataset, eval_dataset, args, timer)
    trainer.train()
        

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
    if args.is_master:
        timer.report("Finished!")
        wandb.finish()
    
