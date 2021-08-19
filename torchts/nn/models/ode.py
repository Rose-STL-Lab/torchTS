import torch
from torch import nn

from torchts.nn.model import TimeSeriesModel


class ODESolver(TimeSeriesModel):
    def __init__(
        self, ode, init_vars, init_coeffs, dt, solver="euler", outvar=None, **kwargs
    ):
        super().__init__(**kwargs)

        if ode.keys() != init_vars.keys():
            raise ValueError("Inconsistent keys in ode and init_vars")

        if solver == "euler":
            self.solver = self.euler
        elif solver == "rk4":
            self.solver = self.runge_kutta_4
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

    def euler(self, nt):
        pred = {name: value.unsqueeze(0) for name, value in self.init_vars.items()}

        for n in range(nt - 1):
            # create dictionary containing values from previous time step
            prev_val = {var: pred[var][[n]] for var in self.var_names}

            for var in self.var_names:
                new_val = prev_val[var] + self.ode[var](prev_val, self.coeffs) * self.dt
                pred[var] = torch.cat([pred[var], new_val])

        # reformat output to contain desired (observed) variables
        return torch.stack([pred[var] for var in self.outvar], dim=1)

    def runge_kutta_4(self, nt):
        pred = {name: value.unsqueeze(0) for name, value in self.init_vars.items()}

        for n in range(nt - 1):
            # create dictionary containing values from previous time step
            prev_val = {var: pred[var][[n]] for var in self.var_names}

            k_1 = prev_val
            k_2 = {}
            k_3 = {}
            k_4 = {}

            for var in self.var_names:
                k_2[var] = (
                    prev_val[var] + self.ode[var](prev_val, self.coeffs) * 0.5 * self.dt
                )

            for var in self.var_names:
                k_3[var] = (
                    prev_val[var] + self.ode[var](k_2, self.coeffs) * 0.5 * self.dt
                )

            for var in self.var_names:
                k_4[var] = prev_val[var] + self.ode[var](k_3, self.coeffs) * self.dt

            for var in self.var_names:
                new_val = (
                    prev_val[var]
                    + (
                        self.ode[var](k_1, self.coeffs) / 6
                        + self.ode[var](k_2, self.coeffs) / 3
                        + self.ode[var](k_3, self.coeffs) / 3
                        + self.ode[var](k_4, self.coeffs) / 6
                    )
                    * self.dt
                )
                pred[var] = torch.cat([pred[var], new_val])

        # reformat output to contain desired (observed) variables
        return torch.stack([pred[var] for var in self.outvar], dim=1)

    def forward(self, nt):
        return self.solver(nt)

    def get_coeffs(self):
        return {name: param.item() for name, param in self.named_parameters()}

    def _step(self, batch, batch_idx, num_batches):
        (x,) = batch
        nt = x.shape[0]
        pred = self(nt)
        return self.criterion(pred, x)