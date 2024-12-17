# utils.py
from torch.utils.data import DataLoader
import einops
from torch import nn

class TransformDataLoader:
    def __init__(self, data, transform_fn, batch_size):
        self.dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.transform_fn = transform_fn

    def __iter__(self):
        for batch in self.dataloader:
            yield self.transform_fn(batch)

    def __len__(self):
        return len(self.dataloader)
    


def HoraShape(tensor, n_dim, n_features):
    return einops.repeat(
        einops.einsum(
            tensor, '... h w -> h ...'), 'h ... -> h d ... r', r=n_features, d=n_dim).shape



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
