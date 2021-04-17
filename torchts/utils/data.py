from collections.abc import Iterable


def lagmat(tensor, lags, horizon=1, dim=0, step=1):
    if isinstance(lags, int):
        data = tensor.unfold(dim, lags + horizon, step)
        x, y = data[:, :lags], data[:, -1]
    elif isinstance(lags, Iterable) and all(isinstance(lag, int) for lag in lags):
        data = tensor.unfold(dim, max(lags) + horizon, step)
        x, y = data[:, [lag - 1 for lag in lags]], data[:, -1]
    else:
        raise TypeError("lags must be of type int or Iterable[int]")

    return x, y
