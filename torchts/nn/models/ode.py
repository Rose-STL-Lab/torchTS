import torch
from torch import nn

from torchts.nn.model import TimeSeriesModel


class ODESolver(TimeSeriesModel):
    def __init__(
        self, ode, init_vars, init_coeffs, dt, solver="euler", outvar=None, **kwargs
    ):
        """
        Args:
            ode (dict): ODE in dictionary form.
            init_vars (dict): Initial values for each variable.
            init_coeffs (dict): Initial values for each parameter.
            dt (float): Time step.
            solver ("euler"/"rk4"): Numerical method for solving the ODE.
            outvar (list): Observed variables.
            kwargs.optimizer (torch.optim): Optimizer.
        """
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

        self.observed = set(self.outvar) == set(
            self.var_names
        )  # Figures out method of training

        self.dt = dt

    def euler_step(self, prev_val):
        """ Computes a single Euler's method step for the ODE
        Args:
            prev_val (dict): Previous values for each variable.
        Returns:
            pred (dict): Euler's method step prediction.
        """
        pred = {name: value.unsqueeze(0) for name, value in self.init_vars.items()}
        for var in self.var_names:
            pred[var] = prev_val[var] + self.ode[var](prev_val, self.coeffs) * self.dt
        return pred

    def runge_kutta_4_step(self, prev_val):
        """ Computes a single 4th order Runge-Kutta method step for the ODE
        Args:
            prev_val (dict): Previous values for each variable.
        Returns:
            pred (dict): 4th order Runge-Kutta method step prediction.
        """
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

    def solver(self, nt, initial=None):
        """ Numerical simulation of the ODE using method self.step_solver
        Args:
            nt (int): Number of time-steps.
            initial (dict): Initial values for each variable.
        Returns:
            pred (dict): Prediction of each variable after nt time steps.
        """
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

        if self.observed:
            # retrieve numerical simulation of single time-steps for each datapoint
            self.zero_grad()
            init_point = {var: x[:, i] for i, var in enumerate(self.outvar)}
            pred = self.step_solver(init_point)
            predictions = torch.stack([pred[var] for var in self.outvar], dim=1)
        else:
            # retrieve numerical simulation of the whole dataset
            nt = x.shape[0]
            predictions = self(nt)

        loss = self.criterion(predictions, y)
        return loss

    def backward(self, loss, optimizer, optimizer_idx):
        # use retain_graph=True to mitigate RuntimeError
        loss.backward(retain_graph=True)
