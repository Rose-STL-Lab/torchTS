import torch
from torch import nn

from torchts.nn.model import TimeSeriesModel


class SimpleAR(TimeSeriesModel):
    def __init__(self, p, bias=True, **kwargs):
        super().__init__(**kwargs)
        self.linear = nn.Linear(p, 1, bias=bias)

    def forward(self, x):
        return self.linear(x)


class MultiAR(TimeSeriesModel):
    def __init__(self, p, k, bias=True, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.k = k
        self.layers = nn.ModuleList(nn.Linear(k, k, bias=False) for _ in range(p))
        self.bias = nn.Parameter(torch.zeros(k)) if bias else None

    def forward(self, x):
        y = torch.zeros(x.shape[0], self.k)

        for i in range(self.p):
            y += self.layers[i](x[:, i, :])

        if self.bias is not None:
            y += self.bias

        return y
