import os
import pickle
from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class TimeSeriesDataset(TensorDataset):
    """
    dataset tensor
    """
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor
        
    def __getitem__(self, index):
        #use thrid column and onwards as sample, may change later
        sample = self.data_tensor[index, 2:]
        #predict third column, may change later
        target = self.data_tensor[index, 2]
        return sample, target
    
    def __len__(self):
        return self.data_tensor.shape[0]

class PaddedDataset(TensorDataset):
    def __init__(self, batch_size, *data, pad_with_last_sample=True):
        data_pad = [None] * len(data)

        if pad_with_last_sample:
            num_padding = (batch_size - (len(data[0]) % batch_size)) % batch_size

            for i in range(len(data)):
                padding = np.repeat(data[i][-1:], num_padding, axis=0)
                data_pad[i] = np.concatenate([data[i], padding], axis=0)

        super().__init__(*(torch.from_numpy(d).float() for d in data_pad))

def standardize(data):
    """
    standardize each dimension
    """
    standardized_data = []
    for sub_array in data:
        sub_array = np.array(sub_array, dtype=object)
        sub_array_standardized = sub_array[:, :2]
        sub_array_standardized = np.append(sub_array_standardized,np.zeros(sub_array[:, 2:].shape),1)
        for i in range(2, sub_array.shape[1]):
            mean = np.mean(sub_array[:, i].astype(float))
            std = np.std(sub_array[:, i].astype(float))
            sub_array_standardized[:, i] = (sub_array[:, i].astype(float) - mean) / std
        standardized_data.append(sub_array_standardized.tolist())
    return standardized_data

def concat(a, b):
    return torch.cat([a, b.unsqueeze(0)], dim=0)


def generate_ode_dataset(x):
    """Generates dataset for ODESolver when training with zero unobserved variables.

    Args:
        x (torch.Tensor): Original time series data

    Returns:
        torch.Tensor: Time series data from time step 0 to n-1
        torch.Tensor: Time series data from time step 1 to n
    """
    n = x.shape[0]
    return x[: n - 1], x[1:]

def load_files(root_dir):
    #I only test it for reading txt
    dataArray = []
    for filename in os.listdir(root_dir):
        file_path = os.path.join(root_dir, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                file_contents = file.read()
                lines = file_contents.strip().split("\n")
                lists = [line.split(",") for line in lines]
                #change string to float, if the string is a number
                dataArray.append([[float(val) if val.replace('.','',1).isdigit() else val for val in sublist] for sublist in lists])
    return dataArray

def split_dataset(root_dir, train_proportion=0.8, test_proportion=0.1, val_proportion=0.1, standardize=None):
    """
    we split the data into train, test and validation by files. For example, we will use data in 1.txt,2.txt,...,8.txt as train
    9.txt as test, 10.txt as val.
    """
    if not isinstance(train_proportion, float) or not isinstance(test_proportion, float) or not isinstance(val_proportion, float):
        raise TypeError("Inputs must be floats.")
    if train_proportion+test_proportion+val_proportion>1:
        raise ValueError("train_proportion + test_proportion + val_proportion must not be greater than 1")
    if train_proportion<=0 or test_proportion<=0:
        raise ValueError("train_proportion or test_proportion must be larger than 0")
    if val_proportion<0:
        raise ValueError("val_proportion is smaller than 0")
    
    dataset = load_files(root_dir)
    
    if standardize:
        dataset = standardize(dataset)
    
    l = len(dataset)
    train = []
    test = []
    valid = []
    for i in range(l):
        if i<=l*train_proportion-1:
            train.append(dataset[i])
        elif i<=l*(train_proportion + test_proportion)-1:
            test.append(dataset[i])
        elif i<=l*(train_proportion + test_proportion + val_proportion)-1:
            valid.append(dataset[i])
        else:
            break
        
    return train,test,valid

def load_dataset(datasets, batch_size, val_batch_size=None, test_batch_size=None, standardize=None):
    loadedData = DataLoader(TimeSeriesDataset(datasets), batch_size=batch_size, shuffle=False)
    return loadedData


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
