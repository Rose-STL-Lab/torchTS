from abc import ABC, abstractmethod

from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class TimeSeriesModel(ABC, nn.Module):
    def __init__(
        self,
        criterion=nn.MSELoss,
        optimizer=optim.SGD,
        lr=1e-2,
        max_epochs=10,
        batch_size=128,
        device="cpu",
    ):
        super().__init__()
        self.criterion = criterion()
        self.optimizer = optimizer
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device

    def fit(self, x, y):
        if not isinstance(self.optimizer, optim.Optimizer):
            self.optimizer = self.optimizer(self.parameters(), lr=self.lr)

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.max_epochs):
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
