---
title: Convolutional LSTM
slug: /convolutional-lstm
---

In spatiotemporal forecasting, assume we have multiple time series generated from a fixed space $x(s,t)$. [Convolutional LSTM](https://papers.nips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf) models the time series on a regular grid, similar to a video.

Convolutional LSTM replaces the matrix multiplication in a regular LSTM with convolution. It determines the future state of a certain cell in the grid by the inputs and past states of its local neighbors:

$$
\begin{bmatrix} i_t \\ f_t \\ o_t \end{bmatrix} = \sigma\big(W^{x} \star x_t + W^h \star h_{t-1} + W^c \circ c_{t-1} + b\big)
$$
