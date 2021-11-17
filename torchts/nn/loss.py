import torch

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()

    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[torch.isnan(loss)] = 0

    return loss.mean()


def quantile_loss(y_pred: torch.tensor, y_true: torch.tensor, quantile: float) -> torch.tensor:
    """Calculate quantile loss

    Args:
        y_pred (torch.tensor): prediction
        y_true (torch.tensor): ground truth
        q (float): quantile

    Returns:
        torch.tensor: output losses
    """
    errors = y_true - y_pred
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    loss = torch.mean(loss)
    return loss