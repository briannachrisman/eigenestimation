import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import List, Any

def MSELoss(x, y):
    return (x - y)**2


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
        preds_log = torch.log(preds)
        truth_log = torch.log(truth)

        # Compute KL divergence per sample (without reduction)
        kl_divergence0 = F.kl_div(truth_log, truth, reduction='none')
        per_sample_kl_div0 = kl_divergence0.sum(dim=-1)  # Sum over classes for each sample

        kl_divergence = F.kl_div(preds_log, truth, reduction='none')
        per_sample_kl_div = kl_divergence.sum(dim=-1) - per_sample_kl_div0  # Sum over classes for each sample
        

        per_sample_kl_div = per_sample_kl_div - per_sample_kl_div0


        # Apply reduction
        if self.reduction == 'mean':
            return per_sample_kl_div.mean(dim=-1)
        elif self.reduction == 'sum':
            return per_sample_kl_div.sum()
        else:  # 'none'
            return per_sample_kl_div