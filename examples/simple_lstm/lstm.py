import torch

from torchts.nn.model import TimeSeriesModel

class LSTM(TimeSeriesModel):
    def __init__(self, input_size, output_size, optimizer, hidden_size=8, batch_size=10, **kwargs):
        super(LSTM, self).__init__(optimizer, **kwargs)
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.lstm = torch.nn.LSTMCell(input_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        return (torch.zeros(self.batch_size, self.hidden_size),
                torch.zeros(self.batch_size, self.hidden_size))

    def forward(self, x, y=None, batches_seen=None):
        hc = self.init_hidden()
        
        hidden, _ = self.lstm(x, hc)
        out = self.linear(hidden)
        return out