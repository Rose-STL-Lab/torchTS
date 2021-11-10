import torch
import numpy as np

from lstm import LSTM

# config
INPUT_SIZE = 1
OUTPUT_SIZE = 1
OPTIMIZER_ARGS = {"lr": 0.01}
BATCH_SIZE = 10


def generate_data():
    # generate linear time series data with some noise
    x = np.linspace(-10, 10, 100).reshape(-1, 1).astype(np.float32)
    y = 2 * x + 1 + np.random.normal(0, 2, x.shape).reshape(-1, 1).astype(np.float32)
    return x, y


def train_model(model, x, y, batch_size):
    # train model
    model.fit(
        torch.from_numpy(x),
        torch.from_numpy(y),
        max_epochs=120,
        batch_size=batch_size,
    )


def main():
    x, y = generate_data()

    criterion = torch.nn.MSELoss()
    model = LSTM(
        INPUT_SIZE,
        OUTPUT_SIZE,
        torch.optim.Adam,
        criterion=criterion,
        optimizer_args=OPTIMIZER_ARGS,
    )

    # train model
    model.fit(
        torch.from_numpy(x),
        torch.from_numpy(y),
        max_epochs=120,
        batch_size=BATCH_SIZE,
    )

    # inference
    y_pred = []
    for x_batch in torch.split(torch.from_numpy(x), BATCH_SIZE):
        y_pred.append(model.predict(x_batch).detach().numpy())
    y_pred = np.concatenate(y_pred)
    return y_pred


if __name__ == "__main__":
    y_pred = main()
    print(y_pred)
