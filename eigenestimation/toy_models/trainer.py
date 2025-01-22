import os
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

class Trainer:
    """
    Trainer class for distributed training of models using PyTorch DDP.

    Args:
        model (torch.nn.Module): The model to train.
        train_data (Dataset): The training dataset.
        eval_data (Dataset): The evaluation dataset.
        args (argparse.Namespace): Training configuration arguments.
    """
    def __init__(self, model, loss_fn, train_data, eval_data, args, timer):
        """
        Initializes the Trainer object by setting up the model, dataloaders, optimizers, and checkpoint.
        """
        self.timer = timer
        self.device_id = args.device_id
        self.is_master = args.is_master
        # Data loaders and samplers
        self.train_sampler = InterruptableDistributedSampler(train_data)
        self.eval_sampler = InterruptableDistributedSampler(eval_data)
        self.train_dataloader = DataLoader(train_data, batch_size=args.batch_size, sampler=self.train_sampler)
        self.eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size, sampler=self.eval_sampler)
        self.timer.report("Data loaders initialized")

        # Model and training utilities setup
        self.model = DDP(model.to(self.device_id), device_ids=[self.device_id])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=.001)
        self.loss_fn = loss_fn
        self.metrics = {"train": MetricsTracker(), "eval": MetricsTracker()}
        self.checkpoint_path = args.checkpoint_path 
        self.checkpoint_epochs = args.checkpoint_epochs
        self.log_epochs = args.log_epochs
        self.epochs = args.epochs

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
            wandb.log({f"{phase}/loss": metrics["loss"], "epoch": epoch})

    def train_one_epoch(self, train_dataloader, epoch):
        """
        Performs one epoch of training, iterating over the training dataset.

        Args:
            epoch (int): The current epoch number.
        """
        self.model.train()
        total_loss = 0
        total_batches = 0
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(self.device_id), targets.to(self.device_id)
    
            # Forward pass
            predictions = self.model(inputs)
            loss = self.loss_fn(predictions, targets)
            total_loss = total_loss + loss
            total_batches = total_batches +1
            # Backpropagation and optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Log to Weights & Biases
        if epoch % self.log_epochs == 0:
            if self.is_master:
                self.log_metrics_to_wandb({"loss": total_loss/total_batches}, epoch, "train")
                self.timer.report(f"Epoch {epoch} Training loss: {total_loss/total_batches}")


    def evaluate(self, epoch):
        """
        Evaluates the model on the evaluation dataset.

        Args:
            epoch (int): The current epoch number.
        """
        self.model.eval()
        loss = 0
        total_batches = 0
        with torch.no_grad():
            for inputs, targets in self.eval_dataloader:
                inputs, targets = inputs.to(self.device_id), targets.to(self.device_id)
                predictions = self.model(inputs)
                loss = loss + self.loss_fn(predictions, targets)
                total_batches += 1

            # Log evaluation metrics
            if self.is_master:
                self.log_metrics_to_wandb({"loss": loss/total_batches}, epoch, "eval")
                self.timer.report(f"Epoch {epoch} Eval loss: {loss/total_batches}")

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
            self.timer.report("Finished training!")
            self.save_checkpoint()
            self.timer.report("Done")

