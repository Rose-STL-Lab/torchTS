import pytest
import torch

from torchts.nn.loss import masked_mae_loss, quantile_loss


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
    "quantile, expected_loss", [(0.05, 0.065), (0.5, 0.05), (0.95, 0.035)]
)
def test_quantile_loss(y_true, y_pred, quantile, expected_loss):
    """Test quantile_loss()"""
    loss = quantile_loss(y_pred, y_true, quantile)
    assert loss == pytest.approx(expected_loss)
