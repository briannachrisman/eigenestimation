# utils.py
from torch.utils.data import DataLoader
import einops
from torch import nn
import wandb
import torch
import os 
import shutil 

class TransformDataLoader:
    def __init__(self, data, transform_fn, batch_size):
        self.dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.transform_fn = transform_fn

    def __iter__(self):
        for batch in self.dataloader:
            yield batch, self.transform_fn(batch)

    def __len__(self):
        return len(self.dataloader)
    




def DeleteParams(model: nn.Module, attributes_to_delete) -> None:
    for attribute_to_delete in attributes_to_delete:
        attribute_list = attribute_to_delete.split('.')
        module = model
        
        # Traverse the attribute hierarchy to find the target parameter
        for attr in attribute_list[:-1]:
            module = getattr(module, attr)

        # Retrieve and delete the parameter, then register it as a buffer
        param = getattr(module, attribute_list[-1])
        delattr(module, attribute_list[-1])
        module.register_buffer(attribute_list[-1], param)
        param.requires_grad = False


def RetrieveWandBArtifact(project_path, metric_name):
    # If you're restoring from a known run:
    api = wandb.Api()

    # Get the artifact if it was logged as an artifact
    artifact = api.artifact(f'{project_path}_{metric_name}:latest')  # Or specify the artifact name and version
    artifact_dir = artifact.download()
    print(artifact_dir)
    # The file would be in artifact_dir, then you can load it:
    restored_artifact = torch.load(os.path.join(artifact_dir, f"{metric_name}.pt"), weights_only=True)


    shutil.rmtree(os.path.dirname(artifact_dir), ignore_errors=True)
    return restored_artifact