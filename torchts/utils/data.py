import os
import pickle
from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class PaddedDataset(TensorDataset):
    def __init__(self, batch_size, *data, pad_with_last_sample=True):
        data_pad = [None] * len(data)

        if pad_with_last_sample:
            num_padding = (batch_size - (len(data[0]) % batch_size)) % batch_size

            for i in range(len(data)):
                padding = np.repeat(data[i][-1:], num_padding, axis=0)
                data_pad[i] = np.concatenate([data[i], padding], axis=0)

        super().__init__(*(torch.from_numpy(d).float() for d in data_pad))


class StandardScaler:
    def __init__(self, data):
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def concat(a, b):
    return torch.cat([a, b.unsqueeze(0)], dim=0)


def load_dataset(dataset_dir, batch_size, val_batch_size=None, test_batch_size=None):
    if val_batch_size is None:
        val_batch_size = batch_size

    if test_batch_size is None:
        test_batch_size = batch_size

    data = {}

    for category in ["train", "val", "test"]:
        cat_data = np.load(os.path.join(dataset_dir, category + ".npz"))
        data["x_" + category] = cat_data["x"]
        data["y_" + category] = cat_data["y"]

    scaler = StandardScaler(data["x_train"][..., 0])

    for category in ["train", "val", "test"]:
        data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])
        data["y_" + category][..., 0] = scaler.transform(data["y_" + category][..., 0])

    data_train = PaddedDataset(batch_size, data["x_train"], data["y_train"])
    data["train_loader"] = DataLoader(data_train, batch_size, shuffle=True)

    data_val = PaddedDataset(val_batch_size, data["x_val"], data["y_val"])
    data["val_loader"] = DataLoader(data_val, val_batch_size, shuffle=False)

    data_test = PaddedDataset(test_batch_size, data["x_test"], data["y_test"])
    data["test_loader"] = DataLoader(data_test, test_batch_size, shuffle=False)

    data["scaler"] = scaler

    return data


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print(f"Unable to load data {pickle_file} : {e}")
        raise e

    return pickle_data


def sliding_window(tensor, lags, horizon=1, dim=0, step=1):
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
