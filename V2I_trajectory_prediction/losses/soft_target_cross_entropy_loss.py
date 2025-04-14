# Credit: https://github.com/ZikangZhou/HiVT
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetCrossEntropyLoss(nn.Module):
    """
    Soft Target Cross Entropy Loss
    Cross Entropy = - sum(p_i * log(q_i))
    """

    def __init__(self, reduction: str = 'mean') -> None:
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(
            self,
            pred: torch.Tensor,  # prediction
            target: torch.Tensor # ground truth
        ) -> torch.Tensor:

        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)

        if self.reduction == 'mean':
            return cross_entropy.mean()
        elif self.reduction == 'sum':
            return cross_entropy.sum()
        elif self.reduction == 'none':
            return cross_entropy
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
