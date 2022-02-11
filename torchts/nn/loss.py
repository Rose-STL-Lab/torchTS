from typing import List, Union

import torch
import numpy as np


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


def quantile_loss(
    y_pred: torch.tensor, y_true: torch.tensor, quantile: Union[float, List[float]]
) -> torch.tensor:
    """Calculate quantile loss

    Args:
        y_pred (torch.tensor): Predicted values
        y_true (torch.tensor): True values
        quantile (float or list): quantile(s) (e.g. 0.5 for median)

    Returns:
        torch.tensor: output losses
    """
    if isinstance(quantile, list):
        errors = torch.repeat_interleave(y_true, len(quantile), dim=1) - y_pred
        quantile = torch.FloatTensor(quantile)
        quantile = quantile.repeat(1, y_true.shape[-1])
    else:
        errors = y_true - y_pred

    loss = torch.max((quantile - 1) * errors, quantile * errors)
    loss = torch.mean(loss, dim=0)
    loss = torch.sum(loss)

    return loss

def quantile_err(prediction, y):
    """
    prediction: arr where first 3 columns are: lower quantile, middle quantile (50%), upper quantile in that order
    """
    y_lower = prediction[:, 0]
    y_upper = prediction[:, 2]
    # Calculate error on our predicted upper and lower quantiles
    # this will get us an array of negative values with the distance between the upper/lower quantile and the
    # 50% quantile
    error_low = y_lower - y
    error_high = y - y_upper
    # Make an array where each entry is the highest error when comparing the upper and lower bounds for that entry prediction
    err = np.maximum(error_high, error_low)
    return err