import torch
import torch.nn as nn
import einops
from typing import List, Any

class TransformerWrapper(nn.Module):
    def __init__(self, transformer: nn.Module, tokenizer: Any) -> None:
        super(TransformerWrapper, self).__init__()
        self.transformer = transformer
        self.tokenizer = tokenizer

    def forward(self, tokenized_X: torch.Tensor) -> torch.Tensor:
        # Generate model outputs
        logits: torch.Tensor = self.transformer(tokenized_X).last_hidden_state
        # Rearrange the output to a flat batch of logits
        return einops.rearrange(logits, 'batch tokens logits -> (batch tokens) logits')

def DeleteParams(model: nn.Module, attributes_to_delete: List[str]) -> None:
    for attribute_to_delete in attributes_to_delete:
        attribute_list: List[str] = attribute_to_delete.split('.')
        module = model
        
        # Traverse the attribute hierarchy to find the target parameter
        for attr in attribute_list[:-1]:
            module = getattr(module, attr)

        # Retrieve and delete the parameter, then register it as a buffer
        param = getattr(module, attribute_list[-1])
        delattr(module, attribute_list[-1])
        module.register_buffer(attribute_list[-1], param)
