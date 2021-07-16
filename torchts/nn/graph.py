import torch
from torch import nn

from torchts.utils import graph
from torchts.utils.data import concat


class DCGRUCell(nn.Module):
    def __init__(
        self,
        num_units,
        adj_mx,
        max_diffusion_step,
        num_nodes,
        input_dim,
        activation=torch.tanh,
        filter_type="laplacian",
        use_gc_for_ru=True,
    ):
        super().__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru

        supports = []

        if filter_type == "laplacian":
            supports.append(graph.scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(graph.random_walk(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(graph.random_walk(adj_mx).T)
            supports.append(graph.reverse_random_walk(adj_mx).T)
        else:
            supports.append(graph.scaled_laplacian(adj_mx))

        for i in range(len(supports)):
            supports[i] = graph.sparse_matrix(supports[i])

        supports = torch.cat([s.unsqueeze(dim=0) for s in supports])
        self.register_buffer("_supports", supports)

        num_matrices = len(supports) * self._max_diffusion_step + 1
        input_size_fc = self._num_units + input_dim
        input_size_gconv = input_size_fc * num_matrices
        input_size_ru = input_size_gconv if self._use_gc_for_ru else input_size_fc

        output_size = 2 * self._num_units
        self._ru_weights = nn.Parameter(torch.empty(input_size_ru, output_size))
        self._ru_biases = nn.Parameter(torch.empty(output_size))
        nn.init.xavier_normal_(self._ru_weights, gain=1.0)
        nn.init.constant_(self._ru_biases, val=1.0)

        output_size = self._num_units
        self._gconv_weights = nn.Parameter(torch.empty(input_size_gconv, output_size))
        self._gconv_biases = nn.Parameter(torch.empty(output_size))
        nn.init.xavier_normal_(self._gconv_weights, gain=1.0)
        nn.init.constant_(self._gconv_biases, val=0.0)

    def forward(self, inputs, hx):
        fn = self._gconv if self._use_gc_for_ru else self._fc
        output_size = 2 * self._num_units
        value = torch.sigmoid(fn(inputs, hx, output_size, bias_start=1.0, reset=True))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))
        c = self._gconv(inputs, r * hx, self._num_units)

        if self._activation is not None:
            c = self._activation(c)

        return u * hx + (1.0 - u) * c

    def _fc(self, inputs, state, output_size, bias_start=0.0, reset=True):
        batch_size = inputs.shape[0]
        shape = (batch_size * self._num_nodes, -1)
        inputs = torch.reshape(inputs, shape)
        state = torch.reshape(state, shape)
        x = torch.cat([inputs, state], dim=-1)

        return torch.matmul(x, self._ru_weights) + self._ru_biases

    def _gconv(self, inputs, state, output_size, bias_start=0.0, reset=False):
        batch_size = inputs.shape[0]
        shape = (batch_size, self._num_nodes, -1)
        inputs = torch.reshape(inputs, shape)
        state = torch.reshape(state, shape)
        x = torch.cat([inputs, state], dim=2)
        input_size = x.size(2)

        x0 = x.permute(1, 2, 0)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step > 0:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = concat(x, x1)

                for _ in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._max_diffusion_step + 1
        x = torch.reshape(
            x, shape=[num_matrices, self._num_nodes, input_size, batch_size]
        )
        x = x.permute(3, 1, 2, 0)
        x = torch.reshape(
            x, shape=[batch_size * self._num_nodes, input_size * num_matrices]
        )

        if reset:
            weights, biases = self._ru_weights, self._ru_biases
        else:
            weights, biases = self._gconv_weights, self._gconv_biases

        x = torch.matmul(x, weights) + biases

        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


class DCGRU(nn.Module):
    def __init__(
        self,
        num_layers,
        num_units,
        adj_mx,
        max_diffusion_step,
        num_nodes,
        input_dim,
        activation=torch.tanh,
        filter_type="laplacian",
        use_gc_for_ru=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            DCGRUCell(
                num_units,
                adj_mx,
                max_diffusion_step,
                num_nodes,
                input_dim if i == 0 else num_units,
                filter_type=filter_type,
                use_gc_for_ru=use_gc_for_ru,
            )
            for i in range(num_layers)
        )

    def forward(self, inputs, hidden_state):
        hidden_states = []
        output = inputs

        for i, layer in enumerate(self.layers):
            output = layer(output, hidden_state[i])
            hidden_states.append(output)

        return output, torch.stack(hidden_states)
