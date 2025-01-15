import os
import torch
import wandb
from torch.utils.data import DataLoader
from pathlib import Path
import argparse


class Trainer:
    def __init__(self, eigenmodel_class, train_data_generator, eval_data_generator, hyperparams, checkpoint_freq, run_name):
        self.eigenmodel_class = eigenmodel_class
        self.train_data_generator = train_data_generator
        self.eval_data_generator = eval_data_generator
        self.hyperparams = hyperparams
        self.checkpoint_freq = checkpoint_freq
        self.run_name = run_name
        self.device = hyperparams.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Initialize W&B
        wandb.init(project="eigenestimation", name=self.run_name, config=self.hyperparams)

        # Prepare model, loss, optimizer, and data
        self.model = self.eigenmodel_class(**hyperparams).to(self.device)
        self.loss_fn = hyperparams.get("loss_fn", torch.nn.MSELoss())
        self.optimizer = hyperparams.get("optimizer", torch.optim.Adam(self.model.parameters(), lr=hyperparams.get("lr", 0.001)))
        self.train_dataloader = self.get_dataloader(train_data_generator, hyperparams["batch_size"])
        self.eval_dataloader = self.get_dataloader(eval_data_generator, hyperparams["batch_size"])

        # Checkpointing via W&B artifacts
        self.start_epoch = self.load_checkpoint_from_wandb()

    def get_dataloader(self, data_generator, batch_size):
        """
        Get DataLoader from the data generator.
        """
        data = data_generator()
        if isinstance(data, DataLoader):
            return data
        else:
            return DataLoader(data, batch_size=batch_size, shuffle=True)

    def save_checkpoint_to_wandb(self, epoch, loss):
        """
        Save checkpoint as a W&B artifact.
        """
        checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss
        }, checkpoint_path)

        # Upload the checkpoint to W&B as an artifact
        artifact = wandb.Artifact(name=f"{self.run_name}_checkpoint", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
        print(f"Checkpoint saved to W&B at epoch {epoch + 1}.")

    def load_checkpoint_from_wandb(self):
        """
        Load checkpoint from W&B if available.
        """
        artifact_name = f"{self.run_name}_checkpoint:latest"
        try:
            artifact = wandb.use_artifact(artifact_name)
            artifact_dir = artifact.download()
            checkpoint_path = os.path.join(artifact_dir, f"checkpoint_epoch.pth")
            checkpoint = torch.load(checkpoint_path)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Resumed training from epoch {checkpoint['epoch'] + 1}")
            return checkpoint["epoch"] + 1
        except wandb.errors.CommError:
            print("No checkpoint found in W&B. Starting from scratch.")
            return 0

    def save_final_artifacts(self):
        """
        Save final model, metrics, and artifacts to W&B.
        """
        # Save the final model to W&B
        final_model_path = "final_model.pth"
        torch.save(self.model.state_dict(), final_model_path)
        final_model_artifact = wandb.Artifact(name=f"{self.run_name}_final_model", type="model")
        final_model_artifact.add_file(final_model_path)
        wandb.log_artifact(final_model_artifact)
        print("Final model saved to W&B!")

        # Save custom artifacts (e.g., activations, plots, etc.)
        metrics_path = "evaluation_metrics.txt"
        with open(metrics_path, "w") as f:
            f.write("Example custom metric: TopActivatingSamples\n")
            f.write("Custom evaluation metrics saved at the end of training.\n")

        evaluation_artifact = wandb.Artifact(name=f"{self.run_name}_evaluation_results", type="evaluation")
        evaluation_artifact.add_file(metrics_path)
        wandb.log_artifact(evaluation_artifact)
        print("Evaluation metrics saved to W&B!")

    def train_loop(self):
        """
        Run the training loop.
        """
        for epoch in range(self.start_epoch, self.hyperparams["epochs"]):
            print(f"Epoch {epoch + 1}/{self.hyperparams['epochs']}")
            self.model.train()
            running_loss = 0.0

            for batch in self.train_dataloader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs["loss"]
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(self.train_dataloader)
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss})

            # Run evaluation
            eval_loss = self.eval_loop(epoch)
            wandb.log({"epoch": epoch, "eval_loss": eval_loss})

            # Save checkpoint to W&B
            if epoch % self.checkpoint_freq == 0 or epoch == self.hyperparams["epochs"] - 1:
                self.save_checkpoint_to_wandb(epoch, avg_train_loss)

        # Save final artifacts at the end of training
        self.save_final_artifacts()

    def eval_loop(self, epoch):
        """
        Run evaluation loop.
        """
        self.model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                eval_loss += outputs["loss"].item()

        avg_eval_loss = eval_loss / len(self.eval_dataloader)
        print(f"Epoch {epoch + 1}: Eval Loss = {avg_eval_loss:.4f}")
        return avg_eval_loss


# ----------------------------------------
# **Main Function to Run Training**
# ----------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Training Script for EigenEstimation")
    parser.add_argument("--checkpoint-freq", type=int, default=5, help="Checkpoint frequency in epochs")
    parser.add_argument("--run-name", type=str, default="eigen_train_run", help="Run name for W&B")
    args = parser.parse_args()

    # Hyperparameters
    hyperparams = {
        "input_dim": 5,
        "hidden_dim": 2,
        "n_networks": 3,
        "hora_features": 15,
        "hora_rank": 1,
        "epochs": 20,
        "batch_size": 32,
        "lr": 0.001,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "loss_fn": torch.nn.MSELoss()
    }

    # Data generators
    def train_data_generator():
        n_datapoints = 3 * 4096
        X_tms_p, _, _ = tms.GenerateTMSDataParallel(
            num_features=5, num_datapoints=n_datapoints, sparsity=0.05, n_networks=3
        )
        return X_tms_p

    def eval_data_generator():
        n_datapoints = 1000
        X_tms_p, _, _ = tms.GenerateTMSDataParallel(
            num_features=5, num_datapoints=n_datapoints, sparsity=0.05, n_networks=3
        )
        return X_tms_p

    # Create Trainer instance
    trainer = Trainer(
        eigenmodel_class=EigenHora,
        train_data_generator=train_data_generator,
        eval_data_generator=eval_data_generator,
        hyperparams=hyperparams,
        checkpoint_freq=args.checkpoint_freq,
        run_name=args.run_name
    )

    # Run Training
    trainer.train_loop()


# ----------------------------------------
# **Run Script**
# ----------------------------------------
if __name__ == "__main__":
    main()
