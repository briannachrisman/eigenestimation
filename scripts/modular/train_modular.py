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

# Append module directory for imports
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eigenestimation"))
sys.path.append(module_dir)

from toy_models.trainer import Trainer
from toy_models.parallel_serial_network import ParallelSerializedModel, CustomMLP


import argparse
from pathlib import Path
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
def get_args_parser():
    """
    Parses command-line arguments for configuring the training process.

    Returns:
        argparse.ArgumentParser: Parsed arguments object.
    """
    parser = argparse.ArgumentParser(description="Training configuration")

    # General training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Directory to save checkpoints")
    parser.add_argument("--subnetwork-path", type=Path, required=True, help="Directory to save subnetwork model")
    parser.add_argument("--checkpoint-epochs", type=int, required=True, help="Frequency at which to save checkpoints")
    parser.add_argument("--log-epochs", type=int, required=True, help="Frequency at which to log metrics")

    # Model and dataset configuration
    parser.add_argument("--n-training-datapoints", type=int, default=100, help="Number of training data points")
    parser.add_argument("--n-eval-datapoints", type=int, default=100, help="Number of evaluation data points")
    parser.add_argument("--wandb-project", type=str, default="subnetworks-fc", help="Weights & Biases project name")

    # Advanced model structure arguments
    parser.add_argument("--max-macrolayers", type=int, default=4, help="Maximum number of macrolayers in the network")
    parser.add_argument("--max-layers-per-mlp", type=int, default=3, help="Maximum number of layers per MLP")
    parser.add_argument("--max-mlps-per-macrolayer", type=int, default=6, help="Maximum number of MLPs per macrolayer")
    parser.add_argument("--max-hidden-units-per-layer", type=int, default=10, help="Maximum hidden units per layer in an MLP")

    # Input and output configuration
    parser.add_argument("--input-dim", type=int, default=10, help="Input dimension for the model")
    parser.add_argument("--output-dim", type=int, default=2, help="Output dimension for the model")

    # Fully connected layer configurations
    parser.add_argument("--n-fc-hidden-units", type=int, default=10, help="Number of hidden units in fully connected layers")
    parser.add_argument("--n-fc-layers", type=int, default=5, help="Number of fully connected layers")

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

    # Create a random set of subnetworks
    mlps_per_macrolayer = [args.input_dim] + [args.max_mlps_per_macrolayer for _ in range(args.max_macrolayers)] + [args.output_dim]
    subnetworks = [
        [
            CustomMLP(mlps_per_macrolayer[i],
                [args.max_hidden_units_per_layer for _ in range(args.max_layers_per_mlp)], 
                1) for _ in range(mlps_per_macrolayer[i+1])
        ] 
        for i in range(len(mlps_per_macrolayer)-1)
    ]

    
    # Use the subnetworks to set up the fully connected model
    subnetwork_model = ParallelSerializedModel(
            parallel_layers=subnetworks,
    ).to(device)
    if args.is_master:
        torch.save(subnetwork_model, args.subnetwork_path)
        
    np.random.seed(42)
    # Generate data
    X_train = 2*torch.rand(args.n_training_datapoints, args.input_dim, device=device)-1  # Batch size of 4, 10 input features
    y_train, _ = subnetwork_model(X_train)
    
    X_eval = 2*torch.rand(args.n_eval_datapoints, args.input_dim, device=device)-1  # Batch size of 4, 10 input features
    y_eval, _ = subnetwork_model(X_eval)
    
    train_dataset = TensorDataset(X_train, y_train.detach())
    eval_dataset = TensorDataset(X_eval, y_eval.detach())
    timer.report("Generated Data")

    fc_network = CustomMLP(
        args.input_dim, [args.n_fc_hidden_units for _ in range(args.n_fc_layers)], args.output_dim)
    
    criterion = nn.MSELoss()
    # Initialize the trainer and start training
    timer.report("Intialized FC Network")
    
    trainer = Trainer(fc_network, criterion, train_dataset, eval_dataset, args, timer)
    trainer.train()
        

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
    if args.is_master:
        timer.report("Finished!")
        wandb.finish()
    
