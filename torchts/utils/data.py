from collections.abc import Iterable


def lagmat(tensor, lags, horizon=1, dim=0, step=1):
    is_int = isinstance(lags, int)
    is_iter = isinstance(lags, Iterable) and all(isinstance(lag, int) for lag in lags)

    if not is_int and not is_iter:
        raise TypeError("lags must be of type int or Iterable[int]")

    if (is_int and lags < 1) or (is_iter and any(lag < 1 for lag in lags)):
        raise ValueError(f"lags must be positive but found {lags}")

    if is_int:
        data = tensor.unfold(dim, lags + horizon, step)
        x, y = data[:, :lags], data[:, -1]
    else:
        data = tensor.unfold(dim, max(lags) + horizon, step)
        x, y = data[:, [lag - 1 for lag in lags]], data[:, -1]

    return x, y
