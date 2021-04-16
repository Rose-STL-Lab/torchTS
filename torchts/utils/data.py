def lagmat(tensor, lags, horizon=1, dim=0, step=1):
    data = tensor.unfold(dim, lags + horizon, step)
    return data[:, :lags], data[:, -1]
