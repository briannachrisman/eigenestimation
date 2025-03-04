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

from eigenestimation.eigenmodel.trainer import Trainer
from eigenestimation.eigenmodel.eigenmodel import EigenModel
from eigenestimation.utils.utils import TransformDataLoader
from eigenestimation.utils.loss import MSELoss, MSEVectorLoss, KLDivergenceSumOverTokensLoss, KLDivergenceFlattenOverTokensLoss

# Ensure correct device usage
device = "cuda" if torch.cuda.is_available() else "cpu"

from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")
from eigenestimation.utils.uniform_models import ZeroOutput

from eigenestimation.eigenmodel.trainer import Trainer
from eigenestimation.eigenmodel.eigenmodel import EigenModel
from eigenestimation.utils.utils import TransformDataLoader, DeleteParams, RetrieveWandBArtifact


from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


from eigenestimation.toy_models.transformer_wrapper import TransformerWrapper
# Ensure correct device usage
device = "cuda" if torch.cuda.is_available() else "cpu"

from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")
from eigenestimation.utils.uniform_models import ZeroOutput


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
    
    parser.add_argument("--params", type=str, default='transformer.transformer.h.5.attn.attention.q_proj.weight,transformer.transformer.h.5.attn.attention.k_proj.weight,transformer.transformer.h.5.attn.attention.v_proj.weight', help="Parameters to keep")
    parser.add_argument("--n-eigenfeatures", type=int, default=2, help="Number of networks")
    
    parser.add_argument("--n-eigenrank", type=int, default=2, help="Number of networks")


    parser.add_argument("--top-k", type=float, default=.1, help="Top k percent of jvp values to keep")
    
    
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


    raw_transformer = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    model = TransformerWrapper(raw_transformer, tokenizer,
                               outputs_logits=False).to(device)

    vals_to_keep = args.params.split(',')
    # Make the eigenestimation a little smaller but only looking at a subset of the parameters.
    # Pick a random subset of tensors to include in paramters, and turn the rest into frozen buffers.
    params_to_delete = [name for name, param in model.named_parameters()]
    params_to_delete = [p for p in params_to_delete if p not in vals_to_keep]

    # Delete 3/4 of the parameters.
    #for p in (params_to_delete[::20]):
    #  params_to_delete.remove(p)

    DeleteParams(model, params_to_delete)
    
    # This is a weird parameter that won't delete right
    #if 'transformer.lm_head.weight' in params_to_delete:
    DeleteParams(model, ['transformer.lm_head.weight'])

    for n,p in model.named_parameters(): print(n, p.shape, p.numel())
    print(sum([p.numel() for p in model.parameters()]), 'parameters')

    # Load in data.
    dataset = load_dataset(args.dataset, split=args.train_split)
    X_train = tokenize_and_concatenate(dataset, model.tokenizer, max_length = token_length, add_bos_token=False)['tokens'].to(device)
    X_train = X_train[torch.randint(len(X_train), (args.n_train_samples,)), :]
    dataset = load_dataset(args.dataset, split=args.eval_split)
    X_eval = tokenize_and_concatenate(dataset, model.tokenizer, max_length = token_length, add_bos_token=False)['tokens'].to(device)
    X_eval = X_eval[torch.randint(len(X_eval), (args.n_eval_samples,)), :]
    
    print('device', device)
    eigenmodel = EigenModel(model, ZeroOutput, KLDivergenceFlattenOverTokensLoss(), 
                            args.n_eigenfeatures, args.n_eigenrank)
    
    
    # Initialize the trainer and start training
    trainer = Trainer(eigenmodel, X_train, X_eval, args, timer)
    trainer.train()

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
    if args.is_master:
        timer.report("Finished!")
        wandb.finish()
