import os
import sys
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from cycling_utils import (
    InterruptableDistributedSampler,
    MetricsTracker,
    atomic_torch_save,
    TimestampedTimer,
)
import wandb  # Add Weights & Biases for logging
import einops 

# Append module directory for imports
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "~/eigenestimation/eigenestimation"))
sys.path.append(module_dir)

from utils.utils import TransformDataLoader

class Trainer:
    """
    Trainer class for distributed training of models using PyTorch DDP.

    Args:
        model (torch.nn.Module): The model to train.
        train_data (Dataset): The training dataset.
        eval_data (Dataset): The evaluation dataset.
        args (argparse.Namespace): Training configuration arguments.
    """
    def __init__(self, eigenmodel, train_data, eval_data, args, timer, compute_gradients=True):
        """
        Initializes the Trainer object by setting up the model, dataloaders, optimizers, and checkpoint.
        """
        self.timer =timer
        self.device_id = args.device_id
        self.is_master = args.is_master
        # Data loaders and samplers
        self.train_sampler = InterruptableDistributedSampler(train_data)
        self.eval_sampler = InterruptableDistributedSampler(eval_data)
        
        self.train_dataloader = DataLoader(train_data, batch_size=args.batch_size, sampler=self.train_sampler)
            
        self.eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size, sampler=self.eval_sampler)
        self.batch_size = args.batch_size
            
        self.timer.report("Data loaders initialized")

        # Model and training utilities setup
        self.model = DDP(eigenmodel.to(self.device_id), device_ids=[self.device_id])
        params_to_optimize = [*[t for name in self.model.module.low_rank_encode for t in self.model.module.low_rank_encode[name]]] + [*[t for name in self.model.module.low_rank_decode for t in self.model.module.low_rank_decode[name]]]
        self.optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step_epochs, gamma=args.lr_decay_rate)

        self.metrics = {"train": MetricsTracker(), "eval": MetricsTracker()}
        self.checkpoint_path = args.checkpoint_path
        self.checkpoint_epochs = args.checkpoint_epochs
        self.log_epochs = args.log_epochs
        self.epochs = args.epochs
        self.L0_penalty = args.L0_penalty
        self.compute_gradients = compute_gradients

        os.makedirs(self.checkpoint_path.parent, exist_ok=True)
        self.timer.report("Model and training utilities initialized")

        # Load checkpoint if exists
        if os.path.isfile(self.checkpoint_path):
            self.load_checkpoint()

    def load_checkpoint(self):
        """
        Loads a training checkpoint to resume training from a saved state.
        """
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=f"cuda:{self.device_id}")
        self.model = DDP(checkpoint["model"].to(self.device_id), device_ids=[self.device_id])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.train_sampler.load_state_dict(checkpoint["train_sampler"])
        self.eval_sampler.load_state_dict(checkpoint["eval_sampler"])
        self.metrics = checkpoint["metrics"]
        self.timer.report("Checkpoint loaded")
        
        
    def save_checkpoint(self):
        """
        Saves the current state of the model, optimizer, sampler, scheduler, and metrics to a checkpoint.
        Also logs the checkpoint to Weights & Biases.
        """
        if self.is_master:
            checkpoint_path = str(self.checkpoint_path)
            atomic_torch_save(
                {
                    "model": self.model.module,
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),

                    "train_sampler": self.train_sampler.state_dict(),
                    "eval_sampler": self.eval_sampler.state_dict(),
                    "metrics": self.metrics,
                },
                self.checkpoint_path,
            )
            self.timer.report("Checkpoint saved")

    def log_metrics_to_wandb(self, metrics, epoch, phase):
        """
        Logs training or evaluation metrics to Weights & Biases.

        Args:
            metrics (dict): Dictionary of metrics to log.
            epoch (int): Current epoch number.
            phase (str): 'train' or 'eval' phase.
        """
        if self.is_master:
            wandb.log({f"{phase}/loss": metrics["loss"], 
                       f"{phase}/reconstruction_loss": metrics["reconstruction_loss"],
                       f"{phase}/sparsity_loss": metrics["sparsity_loss"], "epoch": epoch})


    def train_one_epoch(self, train_dataloader, epoch):
        """
        Performs one epoch of training, iterating over the training dataset.

        Args:
            epoch (int): The current epoch number.
        """
        self.model.train()
        
        
        total_loss = 0
        total_batches = 0
        sparsity_loss = 0
        reconstruction_loss = 0
        baseline_reconstruction_loss = 0
        train_batches_per_epoch = len(train_dataloader)

        for x in train_dataloader:
            # Forward pass
            if self.compute_gradients:
                gradients = self.model.module.compute_gradients(x.to(self.device_id))
            else: gradients = x
            jvp = self.model(gradients)
            reconstruction = self.model.module.reconstruct(jvp)
            batch_shape = ' '.join(["d" + str(i) for i in range(len(jvp.shape)-1)])

                
            #L2_error = torch.stack([
            #    einops.rearrange((reconstruction[name] - gradients[name])**2, #f'{batch_shape} ... -> {batch_shape} (...)') 
            #    for name in gradients
            #], dim=0).mean()
            
            baseline_L2_error = torch.concat([
                ((gradients[name])**2).flatten()
                for name in gradients
            ], dim=0).mean()
            
            L2_error_old = torch.concat([
                ((reconstruction[name] - gradients[name])**2).flatten()
                for name in gradients
            ], dim=0).mean() #/ baseline_L2_error
            
            
            eps = 1e-5
            A_dot_A = (torch.concat([einops.rearrange(
                    reconstruction[name] * reconstruction[name], 
                    f'{batch_shape} ... -> {batch_shape} (...)')
                for name in gradients
            ], dim=-1).sum(dim=-1)) # batch x params#.mean()#sum(dim=0).mean()
            
            B_dot_B = (torch.concat([einops.rearrange(
                    gradients[name] * gradients[name], 
                    f'{batch_shape} ... -> {batch_shape} (...)')
                for name in gradients
            ], dim=-1).sum(dim=-1)) # batch x params#.mean()#sum(dim=0).mean()(dim=0).mean()
            
            A_dot_B = (torch.concat([einops.rearrange(
                    reconstruction[name] * gradients[name], 
                    f'{batch_shape} ... -> {batch_shape} (...)')
                for name in gradients
            ], dim=-1).sum(dim=-1)) # batch x params#.mean()#sum(dim=0).mean()(dim=0).mean()
            
            L2_error = (A_dot_A*A_dot_A - 2*A_dot_B*A_dot_B + B_dot_B*B_dot_B + eps).sqrt().mean()

            
            # TODO (sparsity) # (Batch x V)^T(Vatcg x V).sum()
            L0_error = (abs(jvp**2).sum(dim=-1)).mean() #/ baseline_L2_error #einops.einsum((abs(jvp)), '... f -> ...').mean()
            L = L2_error + self.L0_penalty * L0_error

            sparsity_loss += L0_error.item()
            reconstruction_loss += L2_error.item()
            baseline_reconstruction_loss += baseline_L2_error.item()
            total_loss += L.item()
            total_batches = total_batches +1
            
            
            # Backpropagation and optimizer step
            self.optimizer.zero_grad()
            L.backward()
            self.optimizer.step()
            
            self.model.module.normalize_low_ranks()
            
        if self.is_master:
            self.lr_scheduler.step() 

        # Log to Weights & Biases
        if epoch % self.log_epochs == 0:
            if self.is_master:
                self.log_metrics_to_wandb({
                    "loss": total_loss/total_batches,
                    "reconstruction_loss": reconstruction_loss/total_batches,
                    "sparsity_loss": sparsity_loss/total_batches,}, epoch, "train")
                self.timer.report(
                    f'''
                    Epoch {epoch},
                    training loss: {total_loss/total_batches},
                    training reconstruction_loss: {reconstruction_loss/total_batches},
                    training sparsity_loss: {sparsity_loss/total_batches},
                    training baseline_reconstruction_loss: {baseline_reconstruction_loss/total_batches}'''
                )

    def evaluate(self, epoch):
        """
        Evaluates the model on the evaluation dataset.

        Args:
            epoch (int): The current epoch number.
        """
        self.model.eval()
        total_loss = 0
        total_batches = 0
        sparsity_loss = 0
        reconstruction_loss = 0
        
        for x in self.eval_dataloader:
            if self.compute_gradients:
                gradients = self.model.module.compute_gradients(x.to(self.device_id))
            else: gradients = x            
            # Forward pass
            gradients = {k: v.to(self.device_id) for k, v in gradients.items()}
            jvp = self.model(gradients)
            reconstruction = self.model.module.reconstruct(jvp)
            jvp_einops_shape = ' '.join(["d" + str(i) for i in range(len(jvp.shape)-1)])

            L2_error = torch.stack([
                einops.einsum((reconstruction[name] - gradients[name])**2, f'{jvp_einops_shape} ... -> {jvp_einops_shape}') 
                for name in gradients
            ], dim=0).sum(dim=0).mean()
            L0_error = einops.einsum((abs(jvp)), '... f -> ...').mean()
            L = L2_error + self.L0_penalty * L0_error

            sparsity_loss += L0_error.item()
            reconstruction_loss += L2_error.item()
            total_loss += L.item()
            total_batches = total_batches +1
            
            

        # Log to Weights & Biases
        if epoch % self.log_epochs == 0:
            if self.is_master:
                self.log_metrics_to_wandb({
                    "loss": total_loss/total_batches,
                    "reconstruction_loss": reconstruction_loss/total_batches,
                    "sparsity_loss": sparsity_loss/total_batches,}, epoch, "eval")
                self.timer.report(
                    f'''
                    Epoch {epoch},
                    eval loss: {total_loss/total_batches},
                    eval reconstruction_loss: {reconstruction_loss/total_batches},
                    eval sparsity_loss: {sparsity_loss/total_batches}'''
                )

    def train(self):
        """
        Runs the full training process for the specified number of epochs.
        Calls `train_one_epoch` for training and `evaluate` periodically.
        """
        for epoch in range(self.train_dataloader.sampler.epoch, self.epochs):
            with self.train_dataloader.sampler.in_epoch(epoch):
                self.train_one_epoch(self.train_dataloader, epoch)
                if epoch % self.log_epochs == 0:
                    with self.eval_dataloader.sampler.in_epoch(epoch):
                        self.evaluate(epoch)
                if epoch % self.checkpoint_epochs == 0:
                    self.save_checkpoint()
        if self.is_master:
            self.save_checkpoint()
