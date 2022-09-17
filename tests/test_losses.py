import torch
from torch.nn.functional import cross_entropy

from image_classification.losses import FocalLoss


def test_focal_loss():
    """
    Focal loss should be the same as cross entropy with gamma=0
    """
    target = torch.tensor([3, 0], dtype=int)
    prediction = torch.tensor([[1, 2, 3, 4], [4, 3, 5, 1]], dtype=float)
    assert torch.allclose(
        FocalLoss(gamma=0, reduction="mean")(prediction, target),
        cross_entropy(prediction, target),
    )
