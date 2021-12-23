import numpy as np
import pytest

from torchts.nn.models.ode import ODESolver


@pytest.fixture
def euler_model():
    # ODE: x'(t) = 2x
    model = ODESolver(
        {"x": lambda prev_val, coeffs: coeffs["alpha"] * prev_val["x"]},
        {"x": 1},
        {"alpha": 2},
        0.1,
        solver="euler",
        optimizer=None,
    )
    preds = model(2)
    return model, preds


@pytest.fixture
def rk4_model():
    # ODE: x'(t) = x
    model = ODESolver(
        {"x": lambda prev_val, coeffs: prev_val["x"]},
        {"x": 1},
        {},
        0.1,
        solver="rk4",
        optimizer=None,
    )
    preds = model(2)
    return model, preds


def test_euler(euler_model):
    """Test Euler's Method"""
    model, preds = euler_model
    assert model.step_solver == model.euler_step
    assert model.get_coeffs() == {"alpha": 2}
    # Approximation for exp(0.2)
    assert abs(preds[1, 0].item() - 1.2) < 1e-6


def test_rk4(rk4_model):
    """Test 4th order Runge-Kutta Method"""
    model, preds = rk4_model
    assert model.step_solver == model.runge_kutta_4_step
    # Approximation for exp(0.1)
    assert abs(preds[1, 0].item() - np.exp(0.1)) < 1e-6
