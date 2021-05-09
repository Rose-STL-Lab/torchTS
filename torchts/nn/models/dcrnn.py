import numpy as np
import torch
from torch import nn, optim

from torchts.nn.graph import DCGRU
from torchts.nn.loss import masked_mae_loss
from torchts.nn.model import TimeSeriesModel


class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get("max_diffusion_step", 2))
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 1000))
        self.filter_type = model_kwargs.get("filter_type", "laplacian")
        self.num_nodes = int(model_kwargs.get("num_nodes", 1))
        self.num_rnn_layers = int(model_kwargs.get("num_rnn_layers", 1))
        self.rnn_units = int(model_kwargs.get("rnn_units"))
        self.use_gc_for_ru = bool(model_kwargs.get("use_gc_for_ru", True))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class Encoder(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get("input_dim", 1))
        self.seq_len = int(model_kwargs.get("seq_len"))
        self.dcgru = DCGRU(
            self.num_rnn_layers,
            self.rnn_units,
            adj_mx,
            self.max_diffusion_step,
            self.num_nodes,
            self.input_dim,
            filter_type=self.filter_type,
            use_gc_for_ru=self.use_gc_for_ru,
        )

    def forward(self, inputs, hidden_state):
        output, hidden = self.dcgru(inputs, hidden_state)
        return output, hidden


class Decoder(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.output_dim = int(model_kwargs.get("output_dim", 1))
        self.horizon = int(model_kwargs.get("horizon", 1))
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru = DCGRU(
            self.num_rnn_layers,
            self.rnn_units,
            adj_mx,
            self.max_diffusion_step,
            self.num_nodes,
            self.output_dim,
            filter_type=self.filter_type,
            use_gc_for_ru=self.use_gc_for_ru,
        )

    def forward(self, inputs, hidden_state):
        output, hidden = self.dcgru(inputs, hidden_state)
        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)
        return output, hidden


class DCRNN(TimeSeriesModel, Seq2SeqAttrs):
    def __init__(self, adj_mx, scaler, **model_kwargs):
        super().__init__(criterion=masked_mae_loss, scaler=scaler)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.encoder_model = Encoder(adj_mx, **model_kwargs)
        self.decoder_model = Decoder(adj_mx, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 1000))
        self.use_curriculum_learning = bool(
            model_kwargs.get("use_curriculum_learning", False)
        )

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps)
        )

    def encoder(self, inputs):
        batch_size = inputs.size(1)
        shape = self.num_rnn_layers, batch_size, self.hidden_state_size
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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01, eps=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20, 30, 40, 50], gamma=0.1
        )
        return [optimizer], [scheduler]
