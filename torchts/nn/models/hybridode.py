import torch
from torch import nn

from torchts.nn.models.ode import ODESolver


class HybridODENet(ODESolver):
    def __init__(
        self,
        ode,
        dnns,
        init_vars,
        init_coeffs,
        dt,
        solver="euler",
        outvar=None,
        **kwargs,
    ):
        super().__init__(ode, init_vars, init_coeffs, dt, solver, outvar, **kwargs)

        if ode.keys() != init_vars.keys():
            raise ValueError("Inconsistent keys in ode and init_vars")

        if solver == "euler":
            self.solver = self.euler
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

    def euler(self, nt):
        pred = {name: value.unsqueeze(0) for name, value in self.init_vars.items()}

        for n in range(nt - 1):
            # create dictionary containing values from previous time step
            prev_val = {var: pred[var][[n]] for var in self.var_names}

            for var in self.var_names:
                new_val = prev_val[var] + self.ode[var](prev_val, self.coeffs, self.dnns) * self.dt
                pred[var] = torch.cat([pred[var], new_val])

        # reformat output to contain desired (observed) variables
        return torch.stack([pred[var] for var in self.outvar], dim=1)
