import argparse
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
import wandb  # Add Weights & Biases for tracking
import os
import sys
import transformer_lens
from transformers import AutoTokenizer
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset

# Append module directory for imports
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eigenestimation"))
sys.path.append(module_dir)

from eigenmodel.trainer import Trainer
from eigenmodel.eigenmodel import EigenModel
from utils.utils import TransformDataLoader
from utils.loss import MSELoss, MSEVectorLoss

from toy_models.tms import SingleHiddenLayerPerceptron, GenerateTMSAdditiveData
# Ensure correct device usage
device = "cuda" if torch.cuda.is_available() else "cpu"

from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")
from utils.uniform_models import ZeroOutput

from eigenmodel.trainer import Trainer
from eigenmodel.eigenmodel import EigenModel
from utils.utils import TransformDataLoader, DeleteParams, RetrieveWandBArtifact
from utils.loss import KLDivergenceVectorLoss


from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


from toy_models.transformer_wrapper import TransformerWrapper
# Ensure correct device usage
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    parser.add_argument("--lr-step-epochs", type=int, default=100, help="Learning rate step epochs")
    parser.add_argument("--lr-decay-rate", type=float, default=0.8, help="Learning rate step factor")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--token-length", type=int, default=8, help="Batch size for training")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Directory to save checkpoints")
    
    
    parser.add_argument("--checkpoint-epochs", type=int, required=True, help="Frequency at which to save checkpoints")
    
    parser.add_argument("--log-epochs", type=int, required=True, help="Frequency at which to save checkpoints")
    
    parser.add_argument("--n-features", type=int, default=5, help="Number of input features")
    parser.add_argument("--n-hidden", type=int, default=2, help="Number of hidden features")
    

    parser.add_argument("--n-eigenfeatures", type=int, default=2, help="Number of networks")
    
    parser.add_argument("--n-eigenrank", type=int, default=2, help="Number of networks")

    parser.add_argument("--n-training-datapoints", type=int, default=100, help="Number of training data points")

    parser.add_argument("--top-k", type=float, default=.1, help="Top k percent of jvp values to keep")
    
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
    token_length = args.token_length

    # @title Import pretrained gpt2 (2 layers)
    # Disable fused kernels (FlashAttention and memory-efficient attention)
    # We have to disable this to compute second-order gradients on transformer models.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    # Ensure the math kernel is enabled (it is True by default)
    torch.backends.cuda.enable_math_sdp(True)


    tinystories_1m = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-1M')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    model = TransformerWrapper(tinystories_1m, tokenizer)


    # Make the eigenestimation a little smaller but only looking at a subset of the parameters.
    # Pick a random subset of tensors to include in paramters, and turn the rest into frozen buffers.
    params_to_delete = [name for name, param in model.named_parameters()]
    params_to_delete = [p for p in params_to_delete if #('blocks.4.attn.W' not in p)]# and ('blocks.6.mlp.W' not in p)]#!='transformer.h.1.ln_2.weight']
    'transformer.blocks.3.attn.W_K' not in p]#!='transformer.h.1.ln_2.weight']

    # Delete 3/4 of the parameters.
    #for p in (params_to_delete[::20]):
    #  params_to_delete.remove(p)

    DeleteParams(model, params_to_delete)

    print(sum([p.numel() for p in model.parameters()]))
    for n,p in model.named_parameters(): print(n, p.shape, p.numel())

    # Load in data.
    dataset = load_dataset('roneneldan/TinyStories', split="validation[:1%]")
    X_transformer = tokenize_and_concatenate(dataset, model.tokenizer, max_length = token_length, add_bos_token=False)['tokens']
    train_dataset = X_transformer[:n_training_datapoints]
    del X_transformer
    
    print("HERE")
    
    dataset = load_dataset('roneneldan/TinyStories', split="validation[:1%]")
    X_transformer = tokenize_and_concatenate(dataset, model.tokenizer, max_length = token_length, add_bos_token=False)['tokens']
    eval_dataset = X_transformer[:n_training_datapoints+n_eval_datapoints]
    del X_transformer

    
    eigenmodel = EigenModel(model, ZeroOutput, KLDivergenceVectorLoss(), 
                            args.n_eigenfeatures, args.n_eigenrank)
    
    
    # Initialize the trainer and start training
    trainer = Trainer(eigenmodel, train_dataset, eval_dataset, args, timer)
    trainer.train()

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
    if args.is_master:
        timer.report("Finished!")
        wandb.finish()
