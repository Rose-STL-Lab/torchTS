import torch


def masked_mae_loss(y_pred, y_true):
    """Calculate masked mean absolute error loss

    Args:
        y_pred (torch.Tensor): Predicted values
        y_true (torch.Tensor): True values

    Returns:
        torch.Tensor: Loss
    """
    mask = (y_true != 0).float()
    mask /= mask.mean()

    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[torch.isnan(loss)] = 0

    return loss.mean()


def quantile_loss(y_pred: torch.tensor, y_true: torch.tensor, quantile: float) -> float:
    """Calculate quantile loss

    Args:
        y_pred (torch.tensor): Predicted values
        y_true (torch.tensor): True values
        quantile (float): quantile (e.g. 0.5 for median)

    Returns:
        float: output losses
    """
    errors = y_true - y_pred
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    loss = torch.mean(loss)
    return loss
