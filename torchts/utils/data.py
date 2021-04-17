from collections.abc import Iterable


def lagmat(tensor, lags, horizon=1, dim=0, step=1):
    if isinstance(lags, Iterable):
        data = tensor.unfold(dim, max(lags) + horizon, step)
        x, y = data[:, [lag - 1 for lag in lags]], data[:, -1]
    else:
        data = tensor.unfold(dim, lags + horizon, step)
        x, y = data[:, :lags], data[:, -1]

    return x, y
