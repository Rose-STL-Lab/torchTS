from functools import partial

import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from torchts.nn.model import TimeSeriesModel


class LinearModel(TimeSeriesModel):
    def __init__(self, slope, intercept, **kwargs):
        super().__init__(**kwargs)
        self.line = nn.Linear(1, 1)
        self.line.weight = nn.Parameter(slope * torch.ones_like(self.line.weight))
        self.line.bias = nn.Parameter(intercept * torch.ones_like(self.line.bias))

    def forward(self, x, y=None, batches_seen=None):
        return self.line(x)


def test_forward():
    slope = 2
    intercept = -1
    model = LinearModel(
        slope, intercept, optimizer=optim.SGD, optimizer_args={"lr": 0.01}
    )

    x = torch.Tensor([-1, 0, 1]).reshape(-1, 1)
    y = slope * x + intercept

    assert (model(x) == y).all()
    assert (model.predict(x) == y).all()


def test_step(mocker):
    slope = 2
    intercept = -1
    quantile = 0.5
    quantile_loss = mocker.MagicMock()
    model = LinearModel(
        slope,
        intercept,
        optimizer=optim.SGD,
        criterion=quantile_loss,
        criterion_args={"quantile": quantile},
    )

    x = torch.Tensor([0]).reshape(-1, 1)
    y = slope * x + intercept
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, batch in enumerate(loader):
        model._step(batch, i, len(loader))
        x, y = batch
        quantile_loss.assert_called_once_with(y, y, quantile=quantile)


def test_train():
    torch.manual_seed(0)

    slope_init = 2
    intercept_init = -1
    optimizer = partial(optim.SGD, lr=0.1)
    model = LinearModel(slope_init, intercept_init, optimizer=optimizer)

    slope_true = 1
    intercept_true = 0
    n = 1000
    x = torch.rand(n, 1)
    y = slope_true * x + intercept_true

    max_epochs = 100
    model.fit(x, y, max_epochs=max_epochs)

    tol = 1e-4
    assert pytest.approx(model.line.weight.detach(), abs=tol) == slope_true
    assert pytest.approx(model.line.bias.detach(), abs=tol) == intercept_true
