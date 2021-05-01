import pytest
import torch

from torchts.utils.data import lagmat


@pytest.fixture
def tensor():
    n = 10
    return torch.IntTensor(range(n))


@pytest.mark.parametrize("lag", [2, 5, [1, 2, 3], {1, 2, 3}, [1, 3, 5]])
@pytest.mark.parametrize("horizon", [1, 2])
def test_shape(tensor, lag, horizon):
    x, y = lagmat(tensor, lag, horizon=horizon)

    if isinstance(lag, int):
        rows = len(tensor) - lag - horizon + 1
        cols = lag
    else:
        rows = len(tensor) - max(lag) - horizon + 1
        cols = len(lag)

    assert len(x.shape) == 2
    assert x.shape[0] == rows
    assert x.shape[1] == cols

    assert len(y.shape) == 1
    assert y.shape[0] == rows


@pytest.mark.parametrize("lag", [2, 5, [1, 2, 3], {1, 2, 3}, [1, 3, 5]])
@pytest.mark.parametrize("horizon", [1, 2])
def test_value(tensor, lag, horizon):
    x, y = lagmat(tensor, lag, horizon=horizon)

    if isinstance(lag, int):
        for i in range(x.shape[0]):
            j = lag + i
            assert (x[i, :] == tensor[i:j]).all()
    else:
        for i in range(x.shape[0]):
            assert (x[i, :] == tensor[[x - 1 + i for x in lag]]).all()

    assert (y - x[:, -1] == horizon).all()


@pytest.mark.parametrize("lag", ["1", 1.0, ["1"], [1, "2", 3], {1, 2.0, 3}])
def test_non_int(tensor, lag):
    with pytest.raises(TypeError):
        lagmat(tensor, lag)


@pytest.mark.parametrize("lag", [-1, 0, [0, 1, 2], {0, 1, 2}, [-1, 1, 2]])
def test_non_positive(tensor, lag):
    with pytest.raises(ValueError):
        lagmat(tensor, lag)
