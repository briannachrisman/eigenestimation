# train.py
import torch
import einops
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from eigenestimation.eigenhora import EigenHora  # Assuming eigenestimation.py is in the same directory
import wandb  # Make sure you have installed wandb: pip install wandb
import os 

eps = 1e-10

def Train(
    eigenmodel: EigenHora,
    jacobian_dataloader: DataLoader,
    lr: float,
    n_epochs: int,
    L0_penalty: float,
    device: str = 'cuda',
    project_name: str = 'eigenestimation', 
    run_name: str = '',
    eval_fns=dict(),
    eval_dataloader=None
) -> None:
    # Initialize W&B run
    for _,j in jacobian_dataloader:
        jac_params_metadata = {name:j[name].shape for name in j}
        break

    wandb.init(project=project_name, name=run_name, config={
        'learning_rate': lr,
        'epochs': n_epochs,
        'L0_penalty': L0_penalty,
        'jacobian_shape':jac_params_metadata,
        'n_features': eigenmodel.n_features
    })

    # Optionally watch model for parameter gradients and values
    wandb.watch(eigenmodel, log='all', log_freq=max(1, round(n_epochs / 10)))

    # Move model to device
    eigenmodel.to(device)

    # Collect parameters to optimize (excluding the model's own parameters)
    params_to_optimize = [*[t for name in eigenmodel.low_rank for t in eigenmodel.low_rank[name]]]
    optimizer: Optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    
    for epoch in range(n_epochs):
        sparsity_losses: float = 0.0
        reconstruction_losses: float = 0.0
        total_losses: float = 0.0
        n_batches: int = 0

        # Set model to training mode if it has any normalization/dropout layers
        eigenmodel.train()

        for _, jacobian in jacobian_dataloader:
            # Move batch to device
            jacobian = {k: v.to(device) for k, v in jacobian.items()}

            n_batches += 1
            jvp = eigenmodel(jacobian)
            reconstruction = eigenmodel.reconstruct(jvp.relu())
            jvp_einops_shape = ' '.join(["d" + str(i) for i in range(len(jvp.shape)-1)])
            L2_error = torch.stack([
                einops.einsum((reconstruction[name] - jacobian[name])**2, f'{jvp_einops_shape} ... -> {jvp_einops_shape}') 
                for name in jacobian
            ], dim=0).sum(dim=0).mean()
            L0_error = einops.einsum((abs(jvp)), '... f -> ...').mean()
            L = L2_error + L0_penalty * L0_error
            
            # Backprop
            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            # Accumulate metrics
            sparsity_losses += L0_error.item()
            reconstruction_losses += L2_error.item()
            total_losses += L.item()

        # Compute averages
        avg_total_loss = total_losses / n_batches
        avg_sparsity_loss = sparsity_losses / n_batches
        avg_reconstruction_loss = reconstruction_losses / n_batches

        # Print progress
        if epoch % max(1, round(n_epochs / 10)) == 0:
            print(
                f'Epoch {epoch} : {avg_total_loss:.3f},  '
                f'Reconstruction Loss: {avg_reconstruction_loss:.3f},  '
                f'Sparsity Loss: {avg_sparsity_loss:.3f}'
            )

        # Log metrics to W&B
        wandb.log({
            'total_loss': avg_total_loss,
            'reconstruction_loss': avg_reconstruction_loss,
            'sparsity_loss': avg_sparsity_loss
        })
        
    # After the training loop finishes
    torch.save(eigenmodel.low_rank, "subnetworks.pt")
    artifact = wandb.Artifact(f"{project_name}_{run_name}_subnetworks", type="model-params")
    artifact.add_file("subnetworks.pt")

    print('evaluating...')
    for fn in eval_fns:
        fn_name = fn.__name__
        print(fn_name)
        torch.save(fn(eigenmodel, eval_dataloader, *eval_fns[fn]), f"{fn_name}.pt")
        artifact = wandb.Artifact(f"{project_name}_{run_name}_{fn_name}", type="eval-metrics")
        artifact.add_file(f"{fn_name}.pt")
        wandb.log_artifact(artifact)    

        
    wandb.finish()
    os.remove('subnetworks.pt')
    for fn in eval_fns:
        os.remove(f"{fn_name}.pt")
    
