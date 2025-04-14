# Credit: https://github.com/ZikangZhou/HiVT
import torch
import torch.nn as nn

class LaplaceNLLLoss(nn.Module):
    """
    Negative Log-Likelihood (NLL) loss based on laplace distribution
    NLL = log(2 * scale) + [abs(target - loc) / (scale)]
    """

    def __init__(
            self,
            eps: float = 1e-6,
            reduction: str = 'mean'
        ) -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
            self,
            pred: torch.Tensor,  # prediction
            target: torch.Tensor # ground truth
        ) -> torch.Tensor:

        loc, scale = pred.chunk(2, dim=-1) # divide the last dimension of prediction into two dimensions (loc and scale)
        scale = scale.clone()

        # --------------------
        with torch.no_grad():
            scale.clamp_(min=self.eps) # limit the minimum value of dimension scale
        # --------------------

        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
