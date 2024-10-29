import torch
import torch.nn as nn
import torch.nn.functional as F
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
        probs = logits.softmax(dim=-1)
        return probs #einops.rearrange(probs, 'batch tokens logits -> (batch tokens) logits')

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
        param.requires_grad = False



class KLDivergenceLoss(nn.Module):
    def __init__(self, reduction: str = 'none') -> None:
        """
        KL Divergence Loss with a structure similar to MSELoss.
        
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default is 'mean'.
        """
        super(KLDivergenceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for KL Divergence Loss.
        
        Args:
            preds (torch.Tensor): Predicted logits or probabilities (not softmaxed).
            truth (torch.Tensor): Target probabilities.

        Returns:
            torch.Tensor: The computed KL Divergence Loss.
        """
        # Convert preds to log-probabilities
        preds_log = torch.log_softmax(preds, dim=-1)

        # Compute KL divergence per sample (without reduction)
        kl_divergence = F.kl_div(preds_log, truth, reduction='none')
        per_sample_kl_div = kl_divergence.sum(dim=-1)  # Sum over classes for each sample

        # Apply reduction
        if self.reduction == 'mean':
            return per_sample_kl_div.mean()
        elif self.reduction == 'sum':
            return per_sample_kl_div.sum()
        else:  # 'none'
            return per_sample_kl_div
