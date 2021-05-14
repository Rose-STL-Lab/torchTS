from abc import abstractmethod
from functools import partial

from pytorch_lightning import LightningModule, Trainer
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_LOSS = nn.MSELoss()
DEFAULT_OPT = partial(optim.SGD, lr=1e-2)


class TimeSeriesModel(LightningModule):
    """Base class for all TorchTS models.

    Args:
        criterion: Loss function
        optimizer (torch.optim.Optimizer): Optimizer
    """

    def __init__(
        self, criterion=DEFAULT_LOSS, optimizer=DEFAULT_OPT, scheduler=None, scaler=None
    ):

        super().__init__()
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

    def fit(self, x, y, max_epochs=10, batch_size=128):
        """Fits model to the given data.

        Args:
            x (torch.Tensor): Input data
            y (torch.Tensor): Output data
            max_epochs (int): Number of training epochs
            batch_size (int): Batch size for torch.utils.data.DataLoader
        """
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        trainer = Trainer(max_epochs=max_epochs)
        trainer.fit(self, loader)

    def prepare_batch(self, batch):
        return batch

    def _step(self, batch, batch_idx, loader):
        """

        Args:
            batch: Output of the torch.utils.data.DataLoader
            batch_idx: Integer displaying index of this batch
            dataset: Data set to use

        Returns: loss for the batch

        """

        x, y = self.prepare_batch(batch)
        if self.training:
            batches_seen = batch_idx + self.current_epoch * len(loader)
        else:
            batches_seen = batch_idx

        pred = self(x, y, batches_seen)

        if self.scaler is not None:
            y = self.scaler.inverse_transform(y)
            pred = self.scaler.inverse_transform(pred)

        loss = self.criterion(pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        """Trains model for one step.

        Args:
            batch (torch.Tensor): Output of the torch.utils.data.DataLoader
            batch_idx (int): Integer displaying index of this batch
        """
        train_loss = self._step(batch, batch_idx, loader=self.train_dataloader())
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        """Validates model for one step.

        Args:
            batch (torch.Tensor): Output of the torch.utils.data.DataLoader
            batch_idx (int): Integer displaying index of this batch
        """
        val_loss = self._step(batch, batch_idx, loader=self.val_dataloader())
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        """Tests model for one step.

        Args:
            batch (torch.Tensor): Output of the torch.utils.data.DataLoader
            batch_idx (int): Integer displaying index of this batch
        """
        test_loss = self._step(batch, batch_idx, loader=self.test_dataloader())
        self.log("test_loss", test_loss)
        return test_loss

    @abstractmethod
    def forward(self, x, y=None, batches_seen=None):
        """Forward pass.

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Predicted data
        """

    def predict(self, x):
        """Runs model inference.

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Predicted data
        """
        return self(x).detach()

    def configure_optimizers(self):
        """Configure optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer
        """
        optimizer = self.optimizer(self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return [optimizer], [scheduler]

        return optimizer
