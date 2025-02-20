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

# Append module directory for imports
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eigenestimation"))
sys.path.append(module_dir)

from eigenmodel.trainer import Trainer
from eigenmodel.eigenmodel import EigenModel
from utils.utils import TransformDataLoader
from utils.loss import MSELoss

from toy_models.parallel_serial_network import CustomMLP, ParallelSerializedModel

# Ensure correct device usage
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")
from utils.uniform_models import ZeroOutput

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
        
    parser.add_argument("--checkpoint-epochs", type=int, required=True, help="Frequency at which to save checkpoints")
    
    parser.add_argument("--log-epochs", type=int, required=True, help="Frequency at which to save checkpoints")
        
    parser.add_argument("--n-networks", type=int, default=2, help="Number of networks")

    parser.add_argument("--n-eigenfeatures", type=int, default=2, help="Number of networks")
    
    parser.add_argument("--n-eigenrank", type=int, default=2, help="Number of networks")

    parser.add_argument("--n-training-datapoints", type=int, default=100, help="Number of training data points")

    parser.add_argument("--L0-penalty", type=float, default=.01, help="Penalty")
    
    parser.add_argument("--n-eval-datapoints", type=int, default=100, help="Number of evaluation data points")
        
    parser.add_argument("--wandb-project", type=str, default="tms-autoencoder-training", help="Weights & Biases project name")


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

    model = CustomMLP(args.input_dim, [4, 4, 2, 4, 4, 2], args.output_dim)
    module_list = [
        [[], list(range(2)), list(range(2)), list(range(1)), [], [], [], []],
        [[], list(range(2,4)), list(range(2,4)), list(range(1,2)), [], [], [], []],
        #[[], list(range(4,6)), list(range(6,9)), list(range(2,3)), [], [], [], []],
        [[], [], [], [], list(range(2)), list(range(2)),list(range(1)), []],
        [[], [], [], [], list(range(2,4)), list(range(2,4)), list(range(1,2)), []],
        #[[], [], [], [], list(range(4,6)), list(range(6,9)), list(range(2,3)), [], []],
    ]

    for module in module_list:
        for layer_i, layer in enumerate(module[:-1]):
                    if len(module[layer_i])==0 or len(module[layer_i+1])==0: continue
                    mask = torch.ones_like(model.layers[layer_i].weight.data)
                    for i in module[layer_i]:
                        mask[:,i] = 0
                        for j in module[layer_i+1]:
                            mask[j,i:] = 0
                            mask[j, i] = 1
                    model.layers[layer_i].weight.data = model.layers[layer_i].weight.data*mask

    subnetwork_model = model.to(args.device_id)
    
    # Hyperparameters and model configuration
    n_training_datapoints = args.n_training_datapoints
    n_eval_datapoints = args.n_eval_datapoints
    
    # Generate training and evaluation data
    X_train = 2*torch.rand(args.n_training_datapoints, args.input_dim)-1
    X_eval = 2*torch.rand(args.n_eval_datapoints, args.input_dim)-1
    
    # Create TensorDatasets for DataLoader compatibility
    train_dataset = X_train
    eval_dataset = X_eval
    
    

    eigenmodel = EigenModel(subnetwork_model, ZeroOutput, MSELoss(), args.n_eigenfeatures, args.n_eigenrank)
    
    # Initialize the trainer and start training
    trainer = Trainer(eigenmodel, train_dataset, eval_dataset, args, timer)
    trainer.train()

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
    if args.is_master:
        timer.report("Finished!")
        wandb.finish()
