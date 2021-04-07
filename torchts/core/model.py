from abc import ABC, abstractmethod
from functools import partial

from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_LOSS = nn.MSELoss()
DEFAULT_OPT = partial(optim.SGD, lr=1e-2)


class TimeSeriesModel(ABC, nn.Module):
    def __init__(
        self,
        criterion=DEFAULT_LOSS,
        optimizer=DEFAULT_OPT,
        device="cpu",
    ):
        super().__init__()
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def fit(self, x, y, max_epochs=10, batch_size=128):
        if not isinstance(self.optimizer, optim.Optimizer):
            self.optimizer = self.optimizer(self.parameters())

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(max_epochs):
            self.fit_step(loader)

    def fit_step(self, loader):
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            prediction = self(x)
            loss = self.criterion(y, prediction)
            loss.backward()
            self.optimizer.step()

    @abstractmethod
    def forward(self, x):
        """Forward pass"""

    def predict(self, x):
        return self(x).detach()
