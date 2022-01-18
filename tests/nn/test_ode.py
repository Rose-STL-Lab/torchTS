import numpy as np
import pytest
import torch

from torchts.nn.models.ode import ODESolver
from torchts.utils.data import generate_ode_dataset


@pytest.fixture
def euler_model():
    # ODE: x'(t) = 2x
    model = ODESolver(
        {"x": lambda prev_val, coeffs: coeffs["alpha"] * prev_val["x"]},
        {"x": 1.0},
        {"alpha": 2.0},
        0.1,
        solver="euler",
        optimizer=torch.optim.Adam,
        optimizer_args={"lr": 0.5},
    )
    preds = model(2)
    return model, preds


@pytest.fixture
def euler_unobserved_model():
    # ODE: x'(t) = 2x
    model = ODESolver(
        {
            "x": lambda prev_val, coeffs: coeffs["alpha"] * prev_val["x"],
            "y": lambda prev_val, coeffs: prev_val["x"],
        },
        {"x": 1.0, "y": 1.0},
        {"alpha": 2.0},
        0.1,
        solver="euler",
        outvar=["x"],
        optimizer=torch.optim.Adam,
        optimizer_args={"lr": 0.5},
    )
    preds = model(2)
    return model, preds


@pytest.fixture
def rk4_model():
    # ODE: x'(t) = x
    model = ODESolver(
        {"x": lambda prev_val, coeffs: prev_val["x"]},
        {"x": 1.0},
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
    assert model.get_coeffs() == {"alpha": 2.0}
    # Approximation for exp(0.2)
    assert abs(preds[1, 0].item() - 1.2) < 1e-6


def test_rk4(rk4_model):
    """Test 4th order Runge-Kutta Method"""
    model, preds = rk4_model
    assert model.step_solver == model.runge_kutta_4_step
    # Approximation for exp(0.1)
    assert abs(preds[1, 0].item() - np.exp(0.1)) < 1e-6


def test_generate_ode_dataset(euler_model):
    """Test the generate_ode_dataset function"""
    model, preds = euler_model
    x1, x2 = generate_ode_dataset(preds)
    assert x1 == preds[:-1, :]
    assert x2 == preds[1:, :]


def test_value_errors():
    """Detects ValueError in the __init__ function"""
    with pytest.raises(ValueError, match="Unrecognized solver .*"):
        model = ODESolver(
            {"x": lambda prev_val, coeffs: coeffs["alpha"] * prev_val["x"]},
            {"x": 1.0},
            {"alpha": 2.0},
            0.1,
            solver="a",
            optimizer=None,
        )
    with pytest.raises(ValueError, match="Inconsistent keys in ode and init_vars"):
        model = ODESolver(
            {"x": lambda prev_val, coeffs: coeffs["alpha"] * prev_val["x"]},
            {"x": 1.0, "y": 2.0},
            {"alpha": 2.0},
            0.1,
            solver="euler",
            optimizer=None,
        )
        assert model.step_solver == model.euler_step


def test_step_backward(euler_model):
    """Test the step and backward function"""
    torch.manual_seed(0)
    batch = torch.Tensor([[1.0]]), torch.Tensor([[1.1]])
    model, preds = euler_model
    loss = model._step(batch, 0, 0)
    assert (loss.item() - (1.2 - 1.1) ** 2) < 1e-6
    model.backward(loss, None, 0)
    model.optimizer(model.parameters()).step()
    coeffs = model.get_coeffs()
    assert coeffs["alpha"] < 2


def test_fit(euler_model):
    """Test the step and backward function"""
    torch.manual_seed(0)
    model, preds = euler_model
    model.fit(torch.Tensor([[1.0]]), torch.Tensor([[1.1]]), max_epochs=1, batch_size=1)
    coeffs = model.get_coeffs()
    assert coeffs["alpha"] < 2


def test_unobserved_step_backward(euler_unobserved_model):
    """Test the step and backward function"""
    torch.manual_seed(0)
    batch = torch.Tensor([[1.0, 1.1]]), torch.Tensor([[1.1, 1.2]])
    model, preds = euler_unobserved_model
    loss = model._step(batch, 0, 0)
    assert preds == torch.Tensor([[1.1, 1.2]])
    assert (loss.item() - ((1.1 - 1.0) ** 2 + (1.2 - 1.1) ** 2) / 2) < 1e-6
    model.backward(loss, None, 0)
    model.optimizer(model.parameters()).step()
    coeffs = model.get_coeffs()
    assert coeffs["alpha"] < 2


def test_unobserved_fit(euler_unobserved_model):
    """Test the step and backward function"""
    torch.manual_seed(0)
    model, preds = euler_unobserved_model
    model.fit(
        torch.Tensor([[1.0, 1.1]]),
        torch.Tensor([[1.1, 1.2]]),
        max_epochs=1,
        batch_size=1,
    )
    coeffs = model.get_coeffs()
    assert coeffs["alpha"] < 2
