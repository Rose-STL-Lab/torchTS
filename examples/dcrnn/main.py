import time

import numpy as np
import torch.optim

from torchts.nn.models.dcrnn import DCRNN
from torchts.utils import data as utils

model_config = {
    "cl_decay_steps": 2000,
    "filter_type": "dual_random_walk",
    "horizon": 12,
    "seq_len": 12,
    "input_dim": 2,
    "max_diffusion_step": 2,
    "num_layers": 2,
    "output_dim": 2,
    "use_curriculum_learning": True,
}

optimizer_args = {"lr": 0.01}

# Code to retrieve the graph in the form of an adjacency matrix.
# This corresponds to the distance between 2 traffic sensors in a traffic network.
# For other applications it can mean anything that defines the adjacency between nodes
# eg. distance between airports of different cities when predicting
# covid infection rate.

graph_pkl_filename = "<Path to graph>"

_, _, adj_mx = utils.load_graph_data(graph_pkl_filename)

num_units = adj_mx.shape[0]

model_config["num_nodes"] = num_units

data = np.load(
    "<Path to training *.npz file>"
)  # Absolute path of train, test, val needed.


def run():
    model = DCRNN(
        adj_mx,
        num_units,
        optimizer=torch.optim.SGD,
        optimizer_args=optimizer_args,
        **model_config
    )
    start = time.time()
    model.fit(
        torch.from_numpy(data["x"].astype("float32")),
        torch.from_numpy(data["y"].astype("float32")),
        max_epochs=10,
        batch_size=8,
    )
    end = time.time() - start
    print("Training time taken %f" % (end - start))


if __name__ == "__main__":
    run()
