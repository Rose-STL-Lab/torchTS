import pytest
import torch

from torchts.nn.loss import masked_mae_loss, mis_loss, quantile_loss


@pytest.fixture
def y_true():
    data = [1, 2, 3]
    return torch.tensor(data)


@pytest.fixture
def y_pred():
    data = [1.1, 1.9, 3.1]
    return torch.tensor(data)


def test_masked_mae_loss(y_true, y_pred):
    """Test masked_mae_loss()"""
    loss = masked_mae_loss(y_pred, y_true)
    assert loss == pytest.approx(0.1)


@pytest.mark.parametrize(
    "lower, upper, interval, expected_loss",
    [
        ([1, 2, 3], [1.1, 2.1, 3.1], 0.8, 0.1),
        ([0.9, 1.9, 2.9], [1.1, 2.1, 3.1], 0.8, 0.2),
        ([0.9, 1.9, 2.9], [1.1, 2.1, 3.1], 0.95, 0.2),
        ([0.7, 1.9, 2.9], [0.9, 2.1, 3.1], 0.8, 1.6 / 3),
        ([0.7, 1.9, 2.9], [0.9, 2.1, 3.1], 0.95, 4.6 / 3),
        ([0.9, 1.9, 3.1], [1.1, 2.1, 3.3], 0.8, 1.6 / 3),
        ([0.9, 1.9, 3.1], [1.1, 2.1, 3.3], 0.95, 4.6 / 3),
    ],
)
def test_mis_loss(y_true, lower, upper, interval, expected_loss):
    """Test quantile_loss()"""
    y_true = y_true.reshape(-1, 1)
    y_pred = torch.transpose(torch.tensor([lower, upper]), 0, 1)
    loss = mis_loss(y_pred, y_true, interval)
    assert loss == pytest.approx(expected_loss)


@pytest.mark.parametrize(
    "quantile, expected_loss", [(0.05, 0.065), (0.5, 0.05), (0.95, 0.035)]
)
def test_quantile_loss(y_true, y_pred, quantile, expected_loss):
    """Test quantile_loss()"""
    loss = quantile_loss(y_pred, y_true, quantile)
    assert loss == pytest.approx(expected_loss)
