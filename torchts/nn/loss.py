import torch


def masked_mae_loss(y_pred: torch.tensor, y_true: torch.tensor) -> torch.tensor:
    """Calculate masked mean absolute error loss

    Args:
        y_pred (torch.Tensor): Predicted values
        y_true (torch.Tensor): True values

    Returns:
        torch.Tensor: output loss
    """
    mask = (y_true != 0).float()
    mask /= mask.mean()

    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[torch.isnan(loss)] = 0

    return loss.mean()


def mis_loss(y_pred: torch.tensor, y_true: torch.tensor, interval: float) -> torch.tensor:
    """Calculate MIS loss

    Args:
        y_pred (torch.tensor): Predicted values
        y_true (torch.tensor): True values
        interval (float): confidence interval (e.g. 0.95 for 95% confidence interval)

    Returns:
        torch.tensor: output loss
    """
    alpha = 1 - interval
    lower = y_pred[:, 0::2]
    upper = y_pred[:, 1::2]

    loss = upper - lower
    loss = torch.max(loss, loss + (2 / alpha) * (lower - y_true))
    loss = torch.max(loss, loss + (2 / alpha) * (y_true - upper))
    loss = torch.mean(loss)

    return loss


def quantile_loss(y_pred: torch.tensor, y_true: torch.tensor, quantile: float) -> torch.tensor:
    """Calculate quantile loss

    Args:
        y_pred (torch.tensor): Predicted values
        y_true (torch.tensor): True values
        quantile (float): quantile (e.g. 0.5 for median)

    Returns:
        torch.tensor: output loss
    """
    assert 0 < quantile < 1, "Quantile must be in (0, 1)"
    errors = y_true - y_pred
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    loss = torch.mean(loss)
    return loss


def log_loss(y_pred: torch.tensor, y_true: torch.tensor) -> torch.tensor:
    """Ensure the predictions are in the range (0, 1)
    Args:
        y_pred (torch.tensor): Predicted values
        y_true (torch.tensor): True values

    Returns:
        torch.tensor: output loss
    """
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    return -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))


def mape_loss(y_pred: torch.tensor, y_true: torch.tensor) -> torch.tensor:
    """Calculate mean absolute percentage loss
    Args:
        y_pred (torch.tensor): Predicted values
        y_true (torch.tensor): True values

    Returns:
        torch.tensor: output loss
    """
    return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100


def smape_loss(y_pred: torch.tensor, y_true: torch.tensor) -> torch.tensor:
    """Calculate symmetric mean absolute percentage loss
    Args:
        y_pred (torch.tensor): Predicted values
        y_true (torch.tensor): True values

    Returns:
        torch.tensor: output loss
    """
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
    return torch.mean(torch.abs(y_true - y_pred) / denominator) * 100
