import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import List, Any

#def MSELoss(x, y):
#    return (x - y)**2

#def ErrorLoss(x, y):
#    return x


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
        

class KLDivergenceVectorLoss(nn.Module):
    def __init__(self, reduction: str = 'none') -> None:
        """
        KL Divergence Loss with a structure similar to MSELoss.
        
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default is 'mean'.
        """
        super(KLDivergenceVectorLoss, self).__init__()
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
        
        # Shuffle all elements of preds
        flat_preds = preds.flatten(0,1).detach()
        rand_selection = flat_preds[torch.randint(flat_preds.shape[0], (flat_preds.shape[0],)),...]
        
        #print(rand_selection[0,0])
        # Reshape rand_selection to match preds shape
        reference = rand_selection.view_as(preds)
        
        #truth = preds[torch.randperm(preds.shape[0]),...].detach()
        # Compute KL divergence per sample (without reduction)
        # = F.kl_div(preds, truth.softmax(dim=-1))
        per_sample_kl_div = F.kl_div(preds, reference.softmax(dim=-1), reduction='none').sum(dim=-1).flatten(0,1)#.mean(dim=-1)#(0,1)#.mean(dim=1)#(dim=-1)  # Sum over classes for each sample
        return per_sample_kl_div
        print(per_sample_kl_div.shape)
        #kl_divergence = F.kl_div(preds_log, truth, reduction='none')
        #per_sample_kl_div = kl_divergence.sum(dim=-1) - per_sample_kl_div0  # Sum over classes for each sample
        

        #per_sample_kl_div = per_sample_kl_div - per_sample_kl_div0


        # Apply reduction
        if self.reduction == 'mean':
            return per_sample_kl_div.mean(dim=-1)
        elif self.reduction == 'sum':
            return per_sample_kl_div.sum()
        else:  # 'none'
            return per_sample_kl_div.sum(dim=1)
        
        
        
class MSELoss(nn.Module):
    def __init__(self, reduction: str = 'none') -> None:
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default is 'mean'.
        """
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """
        """
        # 
        per_samples_MSE = ((preds - truth)**2).sum(dim=-1)


        # Apply reduction
        if self.reduction == 'mean':
            return per_samples_MSE.mean(dim=0)
        else:  # 'none'
            return per_samples_MSE
        

        
        
        
   
class MSEVectorLoss(nn.Module):
    def __init__(self, reduction: str = 'none') -> None:
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default is 'mean'.
        """
        super(MSEVectorLoss, self).__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """
        """
        eps = 1e-10
        
        shuffle_indices = torch.randperm(preds.size(0))
        #random_v = torch.where(torch.randn_like(preds) > 0, 1.0, -1.0)
        #print(random_v)
        random_v = (
            (preds - preds[torch.randint(preds.shape[0], (preds.shape[0],)),...].detach()))#/
        #     #(preds.var(dim=0, keepdim=True) + eps)).detach() 
        #random_v = preds.mean(dim=0, keepdim=True)
        #random_v = torch.where(torch.randn_like(preds) > 0, 1.0, -1.0) * ((preds**2).sum(dim=-1, keepdim=True)+eps)**-.5
        
        #random_v = torch.where(torch.randn_like(preds) > 0, 1.0, -1.0)
        #random_v = preds.mean(dim=0, keepdim=True).detach()
        per_samples_MSE = (preds * random_v).mean(dim=-1)


        # Apply reduction
        if self.reduction == 'mean':
            return per_samples_MSE.mean(dim=0)
        else:  # 'none'
            return per_samples_MSE
        
class MSEOutputLoss(nn.Module):
    def __init__(self, reduction: str = 'none') -> None:
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default is 'mean'.
        """
        super(MSEOutputLoss, self).__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """
        """
        # 
        preds_mean = preds.mean(dim=0, keepdim=True)
        per_samples_MSE = (preds * preds.detach()).sum(dim=-1)


        # Apply reduction
        if self.reduction == 'mean':
            return per_samples_MSE.mean(dim=0)
        else:  # 'none'
            return per_samples_MSE
        
        

class ErrorLoss(nn.Module):
    def __init__(self, reduction: str = 'none') -> None:
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default is 'mean'.
        """
        super(ErrorLoss, self).__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """
        """
        # 
        per_samples_MSE = ((preds - truth)).sum(dim=-1)


        # Apply reduction
        if self.reduction == 'mean':
            return per_samples_MSE.mean(dim=0)
        else:  # 'none'
            return per_samples_MSE