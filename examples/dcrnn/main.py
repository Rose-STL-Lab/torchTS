import time

from pytorch_lightning import Trainer

from torchts.nn.models.dcrnn import DCRNN
from torchts.utils import data as utils

data_config = {
    "batch_size": 8,
    "dataset_dir": ".",  # Absolute path to train , test, val expected
    "test_batch_size": 8,
    "val_batch_size": 8,
    "graph_pkl_filename": "adjacency_matrix.pkl",
    # Absolute path to graph file expected
}

model_config = {
    "cl_decay_steps": 2000,
    "filter_type": "dual_random_walk",
    "horizon": 12,
    "input_dim": 2,
    "l1_decay": 0,
    "max_diffusion_step": 2,
    "num_nodes": 320,
    "num_rnn_layers": 2,
    "output_dim": 1,
    "rnn_units": 128,
    "seq_len": 12,
    "use_curriculum_learning": "true",
}

# Code to retrieve the graph in the form of an adjacency matrix.
# This corresponds to the distance between 2 traffic sensors in a traffic network.
# For other applications it can mean anything that defines the adjacency between nodes
# eg. distance between airports of different cities when predicting
# covid infection rate.

graph_pkl_filename = data_config["graph_pkl_filename"]
sensor_ids, sensor_id_to_ind, adj_mx = utils.load_graph_data(graph_pkl_filename)

data = utils.load_dataset(**data_config)
scaler = data["scaler"]

model = DCRNN(adj_mx, scaler, **model_config)


def run():
    trainer = Trainer(max_epochs=10, logger=True)

    start = time.time()
    trainer.fit(model, data["train_loader"], data["val_loader"])
    end = time.time() - start
    print("Training time taken %f" % (end - start))


if __name__ == "__main__":
    run()
