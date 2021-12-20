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
            result = self.ode[var](k_1, self.coeffs) / 6
            result += self.ode[var](k_2, self.coeffs) / 3
            result += self.ode[var](k_3, self.coeffs) / 3
            result += self.ode[var](k_4, self.coeffs) / 6
            pred[var] = prev_val[var] + result * self.dt

        return pred

    # Placeholder fit function. Still trying to figure out why fit() from pytorch_lightning.Trainer() isn't working.
    def fit(
        self,
        x,
        max_epochs=10,
        batch_size=64,
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

        optimizer = self.optimizer(self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
        else:
            scheduler = None

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
                    var: single_point[0, i] for i, var in enumerate(self.outvar)
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
                scheduler.step()

            print("Epoch: " + str(epoch) + "\t Loss: " + str(loss))
            print(self.coeffs)

    def solver(self, nt, initial=None):
        if initial is None:
            initial = self.init_vars
        pred = {name: value.unsqueeze(0) for name, value in initial.items()}

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

    def _step(self, batch, batch_idx, num_batches):
        x, y = self.prepare_batch(batch)
        nt = y.shape[1]

        init_point = {var: x[0, i] for i, var in enumerate(self.var_names)}
        pred = self.solver(nt, init_point)

        # TODO: Account for batch size > 1
        pred = pred.view(1, pred.shape[0], pred.shape[1])

        if self.criterion_args is not None:
            loss = self.criterion(pred, y, **self.criterion_args)
        else:
            loss = self.criterion(pred, y)
        return loss
