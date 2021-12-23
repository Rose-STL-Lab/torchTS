import pytest
import torch

from torchts.nn.models.ode import ODESolver


@pytest.fixture
def euler_model():
    model = ODESolver(
        {"x": lambda prev_val,coeffs: prev_val["x"]}, {"x": 1}, {}, 0.1, solver="euler", optimizer=None
    )
    preds = model(2)
    return model, preds


@pytest.fixture
def rk4_model():
    model = ODESolver(
        {"x": lambda prev_val,coeffs: prev_val["x"]}, {"x": 1}, {}, 0.1, solver="rk4", optimizer=None
    )
    preds = model(2)
    return model, preds


def test_euler(euler_model):
    """Test Euler's Method"""
    model, preds = euler_model
    assert model.step_solver == model.euler_step
    assert abs(preds[1, 0].item() - 1.1) < 1e-6


def test_rk4(rk4_model):
    """Test 4th order Runge-Kutta Method"""
    model, preds = rk4_model
    assert model.step_solver == model.runge_kutta_4_step
    assert abs((preds[1, 0] - torch.exp(1.1)).item()) < 1e-6
