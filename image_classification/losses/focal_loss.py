from typing import Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn


def focal_loss(
    prediction: torch.Tensor,
    label: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Focal loss for multi-class classification.

    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection, 130(4), 485–491.
    https://doi.org/10.1016/j.ajodo.2005.02.022

    Based on https://github.com/zhezh/focalloss/blob/master/focalloss.py

    :param prediction: unnormalized prediction from network
    :param label: ground truth label
    :param class_weights: alpha in paper. Weights to balance classes.
    :param gamma: focusing parameter.
    :param reduction: method to reduce batch loss, mean or sum
    """

    eps = 1e-8
    prediction = F.softmax(prediction, dim=1) + eps
    label_ohe = torch.nn.functional.one_hot(
        label, num_classes=prediction.shape[1]
    ).to(float)
    weight = torch.pow(-prediction + 1.0, gamma)

    if class_weights is None:
        class_weights = torch.ones_like(weight)

    focal = -class_weights * weight * torch.log(prediction)
    loss = torch.sum(label_ohe * focal, dim=1)

    if reduction == "none":
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
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2,
        reduction: str = "mean",
    ):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, prediction: Tensor, label: Tensor):
        return focal_loss(
            prediction,
            label,
            self.class_weights,
            self.gamma,
            self.reduction,
        )
