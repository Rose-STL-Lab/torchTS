import torch
from torch import nn

from torchts.nn.models.ode import ODESolver


class HybridODENet(ODESolver):
    def __init__(
        self, ode, dnns, init_vars, init_coeffs, dt, solver="euler", outvar=None, **kwargs
    ):
        super().__init__(ode, init_vars, init_coeffs, dt, solver, outvar, **kwargs)

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
        self.dnns = dnns
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
            pred[var] = prev_val[var] + self.ode[var](prev_val, self.coeffs, self.dnns) * self.dt
        return pred

    def runge_kutta_4_step(self, prev_val):
        pred = {name: value.unsqueeze(0) for name, value in self.init_vars.items()}

        k_1 = prev_val
        k_2 = {}
        k_3 = {}
        k_4 = {}

        for var in self.var_names:
            k_2[var] = (
                prev_val[var] + self.ode[var](prev_val, self.coeffs, self.dnns) * 0.5 * self.dt
            )

        for var in self.var_names:
            k_3[var] = prev_val[var] + self.ode[var](k_2, self.coeffs, self.dnns) * 0.5 * self.dt

        for var in self.var_names:
            k_4[var] = prev_val[var] + self.ode[var](k_3, self.coeffs, self.dnns) * self.dt

        for var in self.var_names:
            result = self.ode[var](k_1, self.coeffs, self.dnns) / 6
            result += self.ode[var](k_2, self.coeffs, self.dnns) / 3
            result += self.ode[var](k_3, self.coeffs, self.dnns) / 3
            result += self.ode[var](k_4, self.coeffs, self.dnns) / 6
            pred[var] = prev_val[var] + result * self.dt

        return pred


