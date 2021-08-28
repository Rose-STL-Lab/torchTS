import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torchts.nn.model import TimeSeriesModel


class ODESolver(TimeSeriesModel):
    def __init__(
        self, ode, init_vars, init_coeffs, dt, solver="euler", outvar=None, **kwargs
    ):
        super().__init__(**kwargs)

        if ode.keys() != init_vars.keys():
            raise ValueError("Inconsistent keys in ode and init_vars")

        if solver == "euler":
            self.step_solver = self.euler_step
        elif solver == "rk4":
            self.step_solver = self.runge_kutta_4_step
        else:
            raise ValueError(f"Unrecognized solver {solver}")

        for name, value in init_coeffs.items():
            self.register_parameter(name, nn.Parameter(torch.tensor(value)))

        self.ode = ode
        self.var_names = ode.keys()
        self.init_vars = {
            name: torch.tensor(value, device=self.device)
            for name, value in init_vars.items()
        }
        self.coeffs = {name: param for name, param in self.named_parameters()}
        self.outvar = self.var_names if outvar is None else outvar
        self.dt = dt

    def euler_step(self, prev_val):
        pred = {name: value.unsqueeze(0) for name, value in self.init_vars.items()}
        for var in self.var_names:
            pred[var] = prev_val[var] + self.ode[var](prev_val, self.coeffs) * self.dt
        return pred

    def runge_kutta_4_step(self, prev_val):
        pred = {name: value.unsqueeze(0) for name, value in self.init_vars.items()}

        k_1 = prev_val
        k_2 = {}
        k_3 = {}
        k_4 = {}

        for var in self.var_names:
            k_2[var] = (
                prev_val[var] + self.ode[var](prev_val, self.coeffs) * 0.5 * self.dt
            )

        for var in self.var_names:
            k_3[var] = prev_val[var] + self.ode[var](k_2, self.coeffs) * 0.5 * self.dt

        for var in self.var_names:
            k_4[var] = prev_val[var] + self.ode[var](k_3, self.coeffs) * self.dt

        for var in self.var_names:
            pred[var] = (
                prev_val[var]
                + (
                    self.ode[var](k_1, self.coeffs) / 6
                    + self.ode[var](k_2, self.coeffs) / 3
                    + self.ode[var](k_3, self.coeffs) / 3
                    + self.ode[var](k_4, self.coeffs) / 6
                )
                * self.dt
            )

        return pred

    def solver(self, nt):
        pred = {name: value.unsqueeze(0) for name, value in self.init_vars.items()}

        for n in range(nt - 1):
            # create dictionary containing values from previous time step
            prev_val = {var: pred[var][[n]] for var in self.var_names}
            new_val = self.step_solver(prev_val)
            for var in self.var_names:
                pred[var] = torch.cat([pred[var], new_val[var]])

        # reformat output to contain desired (observed) variables
        return torch.stack([pred[var] for var in self.outvar], dim=1)

    def forward(self, nt):
        return self.solver(nt)

    def get_coeffs(self):
        return {name: param.item() for name, param in self.named_parameters()}

    def fit(
        self,
        x,
        optim,
        optim_params=None,
        max_epochs=10,
        scheduler=None,
        scheduler_params=None,
    ):
        """Fits model to the given data by comparing the whole dataset

        Args:
            x (torch.Tensor): Original time series data
            optim (torch.optim): Optimizer
            optim_params: Optimizer parameters
            max_epochs (int): Number of training epochs
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler
            scheduler_params: Learning rate scheduler parameters
        """
        dataset = TensorDataset(x)
        n = dataset.__len__()
        loader = DataLoader(dataset, batch_size=n)

        if optim_params is not None:
            optimizer = optim(self.parameters(), **optim_params)
        else:
            optimizer = optim(self.parameters())

        if scheduler is not None:
            if scheduler_params is not None:
                lr_scheduler = scheduler(optimizer, **scheduler_params)
            else:
                lr_scheduler = scheduler(optimizer)

        for epoch in range(max_epochs):
            for i, data in enumerate(loader, 0):
                self.zero_grad()
                loss = self._step(data, i, n)
                loss.backward(retain_graph=True)
                optimizer.step()
            if scheduler is not None:
                lr_scheduler.step()

            print("Epoch: " + str(epoch) + "\t Loss: " + str(loss))
            print(self.coeffs)

    def fit_random_sample(
        self,
        x,
        optim,
        optim_params=None,
        max_epochs=10,
        batch_size=64,
        scheduler=None,
        scheduler_params=None,
    ):
        """Fits model to the given data by using random samples for each batch

        Args:
            x (torch.Tensor): Original time series data
            optim (torch.optim): Optimizer
            optim_params: Optimizer parameters
            max_epochs (int): Number of training epochs
            batch_size (int): Batch size for torch.utils.data.DataLoader
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler
            scheduler_params: Learning rate scheduler parameters
        """
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size)

        if optim_params is not None:
            optimizer = optim(self.parameters(), **optim_params)
        else:
            optimizer = optim(self.parameters())

        if scheduler is not None:
            if scheduler_params is not None:
                lr_scheduler = scheduler(optimizer, **scheduler_params)
            else:
                lr_scheduler = scheduler(optimizer)

        for epoch in range(max_epochs):
            for i, data in enumerate(loader, 0):
                self.zero_grad()

                n = data[0].shape[0]

                if n < 3:
                    continue

                # Takes a random data point from "data"
                ri = torch.randint(low=0, high=n - 2, size=()).item()
                single_point = data[0][ri : ri + 1, :]
                init_point = {
                    var: single_point[0, i] for i, var in enumerate(self.var_names)
                }

                pred = {
                    name: value.unsqueeze(0) for name, value in self.init_vars.items()
                }

                pred = self.step_solver(init_point)

                predictions = torch.stack([pred[var] for var in self.outvar], dim=0)

                # Compare numerical integration data with next data point
                loss = self.criterion(predictions, data[0][ri + 1, :])

                loss.backward(retain_graph=True)
                optimizer.step()

            if scheduler is not None:
                lr_scheduler.step()

            print("Epoch: " + str(epoch) + "\t Loss: " + str(loss))
            print(self.coeffs)

    def _step(self, batch, batch_idx, num_batches):
        (x,) = batch
        nt = x.shape[0]
        pred = self(nt)
        return self.criterion(pred, x)
