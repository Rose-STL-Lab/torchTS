import numpy as np
import torch
from torch import nn

from torchts.nn.graph import DCGRU
from torchts.nn.model import TimeSeriesModel


class Encoder(nn.Module):
    def __init__(self, input_dim, seq_len, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.dcgru = DCGRU(input_dim=self.input_dim, **kwargs)

    def forward(self, inputs, hidden_state):
        output, hidden = self.dcgru(inputs, hidden_state)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, horizon, **kwargs):
        super().__init__()
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_nodes = kwargs["num_nodes"]
        self.num_units = kwargs["num_units"]
        self.dcgru = DCGRU(input_dim=self.output_dim, **kwargs)
        self.projection_layer = nn.Linear(self.num_units, self.output_dim)

    def forward(self, inputs, hidden_state):
        output, hidden = self.dcgru(inputs, hidden_state)
        projected = self.projection_layer(output.view(-1, self.num_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)
        return output, hidden


class DCRNN(TimeSeriesModel):
    def __init__(
        self,
        adj_mx,
        num_units,
        seq_len=1,
        horizon=1,
        input_dim=1,
        output_dim=1,
        max_diffusion_step=2,
        filter_type="laplacian",
        num_nodes=1,
        num_layers=1,
        use_gc_for_ru=True,
        use_curriculum_learning=False,
        cl_decay_steps=1000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        dcgru_args = {
            "adj_mx": adj_mx,
            "num_nodes": num_nodes,
            "num_layers": num_layers,
            "num_units": num_units,
            "max_diffusion_step": max_diffusion_step,
            "filter_type": filter_type,
            "use_gc_for_ru": use_gc_for_ru,
        }

        self.encoder_model = Encoder(input_dim, seq_len, **dcgru_args)
        self.decoder_model = Decoder(output_dim, horizon, **dcgru_args)

        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.hidden_state_size = num_nodes * num_units
        self.use_curriculum_learning = use_curriculum_learning
        self.cl_decay_steps = cl_decay_steps

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps)
        )

    def encoder(self, inputs):
        batch_size = inputs.size(1)
        shape = self.num_layers, batch_size, self.hidden_state_size
        encoder_hidden_state = torch.zeros(shape, device=self.device)

        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(
                inputs[t], encoder_hidden_state
            )

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        shape = batch_size, self.num_nodes * self.decoder_model.output_dim
        go_symbol = torch.zeros(shape, device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol
        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(
                decoder_input, decoder_hidden_state
            )
            decoder_input = decoder_output
            outputs.append(decoder_output)

            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)

                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]

        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        encoder_hidden_state = self.encoder(inputs)
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)

        return outputs

    def prepare_batch(self, batch):
        x, y = batch
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)

        batch_size = x.size(1)
        x = x.view(
            self.encoder_model.seq_len,
            batch_size,
            self.num_nodes * self.encoder_model.input_dim,
        )
        y = y[..., : self.decoder_model.output_dim].view(
            self.decoder_model.horizon,
            batch_size,
            self.num_nodes * self.decoder_model.output_dim,
        )

        return x, y
