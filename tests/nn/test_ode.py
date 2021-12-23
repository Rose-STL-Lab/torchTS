import pytest
import torch

from torchts.nn.models.ode import ODESolver


def test_euler(y_true, y_pred):
    """Test Euler's Method"""
    model = ODESolver({"x": lambda x: x}, {"x": 1}, {}, 0.1, solver="euler", optimizer=None)
    assert model.step_solver == model.euler_step
    preds = model(1)
    assert preds[1,0].item() == 1.1

def test_rk4(y_true, y_pred):
    """Test 4th order Runge-Kutta Method"""
    model = ODESolver({"x": lambda x: x}, {"x": 1}, {}, 0.1, solver="rk4", optimizer=None)
    assert model.step_solver == model.runge_kutta_4_step
    preds = model(1)
    assert preds[1,0].item() == 1.1
    