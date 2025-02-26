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


from eigenestimation.eigenmodel.trainer import Trainer
from eigenestimation.eigenmodel.eigenmodel import EigenModel
from eigenestimation.utils.utils import TransformDataLoader
from eigenestimation.utils.loss import MSEVectorLoss

from eigenestimation.toy_models.parallel_serial_network import CustomMLP, ParallelSerializedModel
from eigenestimation.toy_models.data import GenerateTMSInputs

# Ensure correct device usage
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")
from eigenestimation.utils.uniform_models import ZeroOutput, MeanOutput
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
    
    parser.add_argument("--lr-step-epochs", type=int, default=100, help="Learning rate")
    parser.add_argument("--lr-decay-rate", type=float, default=0.8, help="Learning rate")
    
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Directory to save checkpoints")

    parser.add_argument("--model-path", type=Path, required=True, help="Directory to save checkpoints")

        
    parser.add_argument("--checkpoint-epochs", type=int, required=True, help="Frequency at which to save checkpoints")
    
    parser.add_argument("--log-epochs", type=int, required=True, help="Frequency at which to save checkpoints")
        
    parser.add_argument("--n-networks", type=int, default=2, help="Number of networks")

    parser.add_argument("--n-eigenfeatures", type=int, default=2, help="Number of networks")
    
    parser.add_argument("--n-eigenrank", type=int, default=2, help="Number of networks")

    parser.add_argument("--n-training-datapoints", type=int, default=100, help="Number of training data points")

    parser.add_argument("--sparsity", type=float, default=.05, help="Sparisty")


    parser.add_argument("--top-k", type=float, default=.1, help="Top k percent of jvp values to keep")
    
    parser.add_argument("--n-eval-datapoints", type=int, default=100, help="Number of evaluation data points")
        
    parser.add_argument("--wandb-project", type=str, default="tms-autoencoder-training", help="Weights & Biases project name")


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


    # Load submodel
    model = torch.load(args.model_path, map_location=f"cuda:{args.device_id}")['model']
    
    
    n_features = model.layers[0].in_features

    # Hyperparameters and model configuration
    n_training_datapoints = args.n_training_datapoints
    n_eval_datapoints = args.n_eval_datapoints
    
    X_train = GenerateTMSInputs(num_features=n_features, num_datapoints=args.n_training_datapoints, sparsity=args.sparsity)
    train_dataset = X_train * (2*torch.rand_like(X_train).round() - 1)
    
    # Make X_train and X_eval not independent
    choice_A = X_train[:,0]
    choice_B = X_train[:,1]
    prob = .5
    # Choose A or B with 50% probability
    choice = torch.rand_like(choice_A)
    X_train[:,0] = (choice < prob) * choice_A + (choice > prob) * choice_B
    
    choice = torch.rand_like(choice_A)
    X_train[:,1] = (choice > prob) * choice_A + (choice < prob) * choice_B
    
    print(X_train.shape)
    
    X_eval = GenerateTMSInputs(num_features=n_features, num_datapoints=args.n_eval_datapoints, sparsity=args.sparsity)
    eval_dataset = X_eval * (2*torch.rand_like(X_eval).round() - 1)

    # Make X_train and X_eval not independent
    choice_A = X_eval[:,0]
    choice_B = X_eval[:,1]
    prob = .5
    # Choose A or B with 50% probability
    choice = torch.rand_like(choice_A)
    X_eval[:,0] = (choice < prob) * choice_A + (choice > prob) * choice_B
    
    choice = torch.rand_like(choice_A)
    X_eval[:,1] = (choice > prob) * choice_A + (choice < prob) * choice_B

    

    eigenmodel = EigenModel(model, MeanOutput, MSEVectorLoss(), args.n_eigenfeatures, args.n_eigenrank)
    
    # Initialize the trainer and start training
    trainer = Trainer(eigenmodel, train_dataset, eval_dataset, args, timer)
    trainer.train()

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
    if args.is_master:
        timer.report("Finished!")
        wandb.finish()
