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
    def __init__(self, model, train_data, eval_data, args):
        """
        Initializes the Trainer object by setting up the model, dataloaders, optimizers, and checkpoint.
        """
        self.timer = TimestampedTimer("Trainer Initialized")
        self.args = args
        self.device_id = args.device_id

        # Data loaders and samplers
        self.train_sampler = InterruptableDistributedSampler(train_data)
        self.eval_sampler = InterruptableDistributedSampler(eval_data)
        self.train_dataloader = DataLoader(train_data, batch_size=args.batch_size, sampler=self.train_sampler)
        self.eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size, sampler=self.eval_sampler)
        self.timer.report("Data loaders initialized")

        # Model and training utilities setup
        self.model = DDP(model.to(self.device_id), device_ids=[self.device_id])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step_epochs, gamma=args.lr_decay_rate)
        self.metrics = {"train": MetricsTracker(), "eval": MetricsTracker()}
        self.checkpoint_path = args.checkpoint_dir / "checkpoint.pt"
        self.checkpoint_epochs = args.checkpoint_epochs
        self.log_epochs = args.log_epochs

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
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_sampler.load_state_dict(checkpoint["train_sampler"])
        self.eval_sampler.load_state_dict(checkpoint["eval_sampler"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.metrics = checkpoint["metrics"]
        self.timer.report("Checkpoint loaded")
        
    def save_checkpoint(self):
        """
        Saves the current state of the model, optimizer, sampler, scheduler, and metrics to a checkpoint.
        Also logs the checkpoint to Weights & Biases.
        """
        if self.args.is_master:
            checkpoint_path = str(self.checkpoint_path)
            atomic_torch_save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "train_sampler": self.train_sampler.state_dict(),
                    "eval_sampler": self.eval_sampler.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
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
        if self.args.is_master:
            wandb.log({f"{phase}/loss": metrics["loss"], f"{phase}/examples_seen": metrics["examples_seen"], "epoch": epoch})

    def train_one_epoch(self, epoch):
        """
        Performs one epoch of training, iterating over the training dataset.

        Args:
            epoch (int): The current epoch number.
        """
        self.model.train()
        for inputs, targets in self.train_dataloader:
            inputs, targets = inputs.to(self.device_id), targets.to(self.device_id)

            # Forward pass
            predictions = self.model(inputs)
            loss = self.loss_fn(predictions, targets)

            # Backpropagation and optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Metrics update and learning rate schedule
        self.metrics["train"].update({"examples_seen": len(inputs), "loss": loss.item()})
        self.metrics["train"].reduce()
        self.lr_scheduler.step()

        # Log to Weights & Biases
        if epoch % self.log_epochs == 0:
            if self.args.is_master:
                total_loss = self.metrics["train"].agg["loss"] / self.metrics["train"].agg["examples_seen"]
                self.log_metrics_to_wandb({"loss": total_loss, "examples_seen": self.metrics["train"].agg["examples_seen"]}, epoch, "train")
                self.timer.report(f"Epoch {epoch} Training loss: {total_loss}")


    def evaluate(self, epoch):
        """
        Evaluates the model on the evaluation dataset.

        Args:
            epoch (int): The current epoch number.
        """
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.eval_dataloader:
                inputs, targets = inputs.to(self.device_id), targets.to(self.device_id)
                predictions = self.model(inputs)
                loss = self.loss_fn(predictions, targets)

                self.metrics["eval"].update({"examples_seen": len(inputs), "loss": loss.item()})
                self.metrics["eval"].reduce()

            # Log evaluation metrics
            if self.args.is_master:
                total_loss = self.metrics["eval"].agg["loss"] / self.metrics["eval"].agg["examples_seen"]
                self.log_metrics_to_wandb({"loss": total_loss, "examples_seen": self.metrics["eval"].agg["examples_seen"]}, epoch, "eval")
                self.timer.report(f"Epoch {epoch} Eval loss: {total_loss}")

    def train(self):
        """
        Runs the full training process for the specified number of epochs.
        Calls `train_one_epoch` for training and `evaluate` periodically.
        """
        for epoch in range(self.train_dataloader.sampler.epoch, self.args.epochs):
            with self.train_dataloader.sampler.in_epoch(epoch):
                self.train_one_epoch(epoch)
                if epoch % self.args.log_epochs == 0:
                    with self.eval_dataloader.sampler.in_epoch(epoch):
                        self.evaluate(epoch)
                if epoch % self.checkpoint_epochs == 0:
                    self.save_checkpoint()
        self.save_checkpoint()

        print("Training Complete!")
