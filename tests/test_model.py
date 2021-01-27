import torch

from torchts.core.model import TimeSeriesModel


class LinearModel(TimeSeriesModel):
    def __init__(self, slope, intercept):
        super().__init__()
        self.line = torch.nn.Linear(1, 1)
        self.line.weight = torch.nn.Parameter(slope * torch.ones_like(self.line.weight))
        self.line.bias = torch.nn.Parameter(intercept * torch.ones_like(self.line.bias))

    def forward(self, x):
        return self.line(x)


def test_model():
    slope = 2
    intercept = -1
    model = LinearModel(slope, intercept)

    x = torch.Tensor([-1, 0, 1]).reshape(-1, 1)
    y = slope * x + intercept

    assert (model(x) == y).all()
    assert (model.predict(x) == y).all()
