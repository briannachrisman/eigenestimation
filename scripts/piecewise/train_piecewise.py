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
from torch import Tensor 
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
    parser.add_argument("--dataset-path", type=Path, required=True, help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-epochs", type=int, required=True, help="Frequency at which to save checkpoints")
    parser.add_argument("--log-epochs", type=int, required=True, help="Frequency at which to log metrics")

    # Model and dataset configuration
    parser.add_argument("--n-training-datapoints", type=int, default=100, help="Number of training data points")
    parser.add_argument("--n-eval-datapoints", type=int, default=100, help="Number of evaluation data points")
    parser.add_argument("--wandb-project", type=str, default="subnetworks-fc", help="Weights & Biases project name")

    # Input and output configuration
    parser.add_argument("--input-dim", type=int, default=10, help="Input dimension for the model")
    parser.add_argument("--output-dim", type=int, default=2, help="Output dimension for the model")

    # Fully connected layer configurations
    parser.add_argument("--n-hidden-units", type=int, default=10, help="Number of hidden units in fully connected layers")
    parser.add_argument("--n-hidden-layers", type=int, default=5, help="Number of fully connected layers")

    parser.add_argument("--n-knots", type=int, default=5, help="Number of fully connected layers")

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

def create_random_piecewise_function(intervals, range_y):
    """
    Create and plot a random piecewise function.

    Parameters:
        intervals (list of tuples): List of (start, end) tuples for each piece.
        range_y (tuple): Range of y-values for the random function.
        x_dense (numpy.ndarray): Dense x-values for the piecewise function.


    Returns:
        y_dense (numpy.ndarray): Corresponding y-values.
    """
    
    # Create random control points for each interval
    x_points = []
    y_points = []
    for start, end in intervals:
        x_points.append(start)
        y_points.append(np.random.uniform(range_y[0], range_y[1]))
    
    # Add the final endpoint
    x_points.append(intervals[-1][1])
    y_points.append(np.random.uniform(range_y[0], range_y[1]))

    x_points = np.array(x_points)
    y_points = np.array(y_points)
    
    return lambda x: np.interp(x, x_points, y_points)

def main(args, timer):
    """
    Main function to initialize data, model, and trainer, and begin training.
    """
    rank = setup_distributed_training(args)
    
    # Initialize Weights & Biases (only in the master process)
    if args.is_master:
        wandb.init(project=args.wandb_project, config=vars(args))
        
    np.random.seed(21)
    # Generate data
    
    
    knots = np.linspace(-1,1, args.n_knots)
    intervals = [(knots[i], knots[i+1]) for i in range(len(knots)-1)]
    range_y = (-1, 1)
    piecewise_fn = create_random_piecewise_function(intervals, range_y)
    
    # Save piecewise_fn as pickled object
    
    X_train = 2*np.random.rand(args.n_training_datapoints, args.input_dim)-1  # Batch size of 4, 10 input features
    y_train = piecewise_fn(X_train)
    
    X_eval = 2*np.random.rand(args.n_training_datapoints, args.input_dim)-1  # Batch size of 4, 10 input features
    y_eval = piecewise_fn(X_eval)
    
    # Save X_train, y_train, X_eval, y_eval as pickled objects
    
    
    train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train))
    if args.is_master:
        torch.save(train_dataset, args.dataset_path)

    eval_dataset = TensorDataset(Tensor(X_eval), Tensor(y_eval))
    timer.report("Generated Data")
    
    print(X_train.shape, y_train.shape, 'SHAPE!!!')
    fc_network = CustomMLP(args.input_dim, [args.n_hidden_units for _ in range(args.n_hidden_layers)], args.output_dim)
    
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
    
