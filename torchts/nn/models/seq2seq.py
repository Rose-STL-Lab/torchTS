import torch
from torch import nn

from torchts.nn.model import TimeSeriesModel


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        """
        Args:
            input_dim: the dimension of input sequences.
            hidden_dim: number hidden units.
            num_layers: number of encode layers.
            dropout_rate: recurrent dropout rate.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout_rate,
            batch_first=True,
        )

    def forward(self, source):
        """
        Args:
            source: input tensor(batch_size*input dimension)
        Return:
            outputs: Prediction
            concat_hidden: hidden states
        """
        outputs, hidden = self.lstm(source)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, dropout_rate):
        """
        Args:
            output_dim: the dimension of output sequences.
            hidden_dim: number hidden units.
            num_layers: number of code layers.
            dropout_rate: recurrent dropout rate.
        """
        super().__init__()

        # Since the encoder is bidirectional, decoder has double hidden size
        self.lstm = nn.LSTM(
            output_dim,
            hidden_dim * 2,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, hidden):
        """
        Args:
            x: prediction from previous prediction.
            hidden: hidden states from previous cell.
        Returns:
            1. prediction for current step.
            2. hidden state pass to next cell.
        """
        output, hidden = self.lstm(x, hidden)
        prediction = self.out(output)
        return prediction, hidden


class Seq2Seq(TimeSeriesModel):
    def __init__(self, encoder, decoder, output_dim, horizon, **kwargs):
        """
        Args:
            encoder: Encoder object.
            decoder: Decoder object.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.output_dim = output_dim
        self.horizon = horizon

    def forward(self, source, target=None, batches_seen=None):
        """
        Args:
            source: input tensor.
        Returns:
            total prediction
        """
        encoder_output, encoder_hidden = self.encoder(source)

        # Concatenate the hidden states of both directions.
        h = torch.cat(
            [
                encoder_hidden[0][0 : self.encoder.num_layers, :, :],
                encoder_hidden[0][-self.encoder.num_layers :, :, :],
            ],
            dim=2,
            out=None,
        )

        c = torch.cat(
            [
                encoder_hidden[1][0 : self.encoder.num_layers, :, :],
                encoder_hidden[1][-self.encoder.num_layers :, :, :],
            ],
            dim=2,
            out=None,
        )

        batch_size = source.size(0)
        shape = (batch_size, 1, self.output_dim)
        decoder_output = torch.zeros(shape, device=self.device)
        decoder_hidden = (h, c)
        outputs = []

        for _ in range(self.horizon):
            decoder_output, decoder_hidden = self.decoder(
                decoder_output, decoder_hidden
            )
            outputs.append(decoder_output)

        return torch.cat(outputs, dim=1)
