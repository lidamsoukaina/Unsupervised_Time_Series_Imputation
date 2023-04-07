import torch
import torch.nn as nn


class ORT_MIT_Loss(nn.Module):
    """Masked Imputation Task (MIT)
    Observed Reconstruction Task (ORT)
    """

    def __init__(self, delta=0.5):
        super().__init__()
        self.delta = delta

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask_artif: torch.BoolTensor,
        mask_orig: torch.BoolTensor,
    ) -> torch.Tensor:
        MIT = torch.sum(torch.square(y_pred - y_true) * mask_artif) / (
            torch.sum(mask_artif)
        )
        mask_total = torch.logical_and(mask_artif, mask_orig)
        ORT = torch.sum(torch.square(y_pred - y_true) * ~mask_total) / (
            torch.sum(~mask_total)
        )
        return ORT + self.delta * MIT


class MaskedMSELoss(nn.Module):
    """Masked MSE Loss"""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.
        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered
        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)


class HuberLossWithMask(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLossWithMask, self).__init__()
        self.delta = delta

    def forward(self, y_true, y_pred, mask):
        error = y_true - y_pred
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        masked_loss = loss * mask.float()
        return masked_loss.mean()
