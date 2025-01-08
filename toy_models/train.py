import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

# Model training function
def TrainModel(
    model: nn.Module,
    criterion: nn.Module,
    learning_rate: float,
    dataloader: DataLoader,
    n_epochs: int,
    device:str
) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
    
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Initialize weights randomly
    for param in model.parameters():
        param.data = torch.randn_like(param)

    # Track losses and parameters
    losses = []
    params = []
    done = False

    # Starting parameters
    params.append(torch.cat([param.data.flatten() for param in model.parameters()]).cpu().numpy())

    for epoch in range(n_epochs):
        if not done:
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()  # Zero the gradients
                outputs = model(x_batch.to(device))  # Forward pass
                loss = criterion(outputs, y_batch.to(device))  # Calculate loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

            # Track progress every 100 epochs after 200
            if epoch % round(n_epochs/10) == 0 and epoch:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
            if epoch > 10:
                if losses[-1] < 0.001:
                    done = True

        # Store loss and parameters
        losses.append(loss.item())
        params.append(torch.cat([param.data.flatten() for param in model.parameters()]).cpu().numpy())

    return model, torch.Tensor(params), torch.Tensor(losses)

# Plot trajectories of loss and parameters
def PlotTrajectories(losses: torch.Tensor, params: torch.Tensor):
    # Earliest index of lowest loss
    final_epoch = max(10, np.argmin(losses.numpy()))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot loss trajectory
    ax[0].plot(losses[1:final_epoch])
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_ylim(0, 1)

    # Normalize parameter trajectories for visualization
    eps = 1e-8
    for i, w in enumerate(params[:final_epoch, :].T):
        ax[1].plot((w - w[0]) / (w[-1] - w[0] + eps), label=f'w{i}')

    # Set labels and title for the parameter plot
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Normalized Weight')
    ax[1].set_title('Parameter Value Trajectories')

    plt.tight_layout()
    plt.show()

# Function to train multiple random models and analyze results
def RandomModels(
    model_type: nn.Module,
    lr: float,
    criterion: nn.Module,
    dataloader: DataLoader,
    n_epochs: int,
    n_iter: int
) -> Tuple[list, list, list]:

    starting_params = []
    ending_params = []
    losses = []

    for i in range(n_iter):
        model = model_type()  # Instantiate a new model
        optimizer = optim.SGD(model.parameters(), lr=lr)
        # Train the model
        model, params, loss = TrainModel(model, criterion, optimizer, dataloader, n_epochs)

        # Record starting and ending parameters and final loss
        starting_params.append(params[0])
        ending_params.append(params[-1])
        losses.append(loss[-1].item())

    return starting_params, ending_params, losses
