from typing import Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn


def focal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
    reduction: Optional[str] = None,
) -> torch.Tensor:
    """
    Focal loss for multi-class classification.

    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection, 130(4), 485–491.
    https://doi.org/10.1016/j.ajodo.2005.02.022

    Based on https://github.com/zhezh/focalloss/blob/master/focalloss.py

    :param prediction: unnormalized prediction from network (N, C)
    :param target: ground truth target, (N,)
    :param class_weights: alpha in paper. Weights to balance classes.
    :param gamma: focusing parameter.
    :param reduction: method to reduce batch loss, mean or sum
    """

    eps = 1e-8

    prediction = F.softmax(prediction, dim=1) + eps

    target_ohe = torch.nn.functional.one_hot(
        target, num_classes=prediction.shape[1]
    )

    weight = torch.pow(-prediction + 1.0, gamma)

    if class_weights is None:
        class_weights = 1

    focal = -class_weights * weight * torch.log(prediction)
    loss = torch.sum(target_ohe * focal, dim=1)

    if reduction is None:
        loss = loss
    elif reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    else:
        raise NotImplementedError(f"Reduction mode {reduction} not available")

    return loss


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.

    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection, 130(4), 485–491.
    https://doi.org/10.1016/j.ajodo.2005.02.022
    """

    def __init__(
        self,
        gamma: float = 2,
        class_weights: Optional[torch.Tensor] = None,
        reduction: Optional[str] = None,
    ):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, prediction: Tensor, target: Tensor):
        return focal_loss(
            prediction,
            target,
            gamma=self.gamma,
            class_weights=self.class_weights,
            reduction=self.reduction,
        )
