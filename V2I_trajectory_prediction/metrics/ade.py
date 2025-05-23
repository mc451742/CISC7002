# Credit: https://github.com/ZikangZhou/HiVT
from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric

class ADE(Metric):
    """
    Minimum Average Displacement Error
    minADE: The average L2 distance between the best forecasted trajectory and the ground truth.
            The best here refers to the trajectory that has the minimum endpoint error.
    """

    def __init__(self,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(ADE, self).__init__(
            dist_sync_on_step=dist_sync_on_step, 
            process_group=process_group, 
            dist_sync_fn=dist_sync_fn
        )
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
        ) -> None:
        
        self.sum += torch.norm(pred - target, p=2, dim=-1).mean(dim=-1).sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
