DCRNN
=====

In spatiotemporal forecasting, assume we have multiple time series generated from a fixed space :math:`x(s,t)`.
`Diffusion Convolutional LSTM <https://openreview.net/pdf?id=SJiHXGWAZ>`_ models the time series on an irregular grid (graph) as a diffusion process.
Diffusion Convolutional LSTM replaces the matrix multiplication in a regular LSTM with diffusion convolution. It determines the future state of a certain cell in the graph by the inputs and past states of its local neighbors:

.. math::
   \begin{bmatrix} i_t \\ f_t \\ o_t \end{bmatrix} = \sigma\big(W^{x} \star_g x_t + W^h \star_g h_{t-1} + W^c \circ c_{t-1} + b\big)


where :math:`W \star_g x = \sum_{i=1}^k \big(D^{-1}A\big)^i \cdot W \cdot x` is the diffusion convolution.

.. automodule:: torchts.nn.models.dcrnn
   :members:
