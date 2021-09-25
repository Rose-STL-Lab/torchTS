---
title: Diffusion Convolutional LSTM
slug: /diffusion-convolutional-lstm
---

In spatiotemporal forecasting, assume we have multiple time series generated from a fixed space $x(s,t)$. [Diffusion Convolutional LSTM](https://openreview.net/pdf?id=SJiHXGWAZ) models the time series on an irregular grid (graph) as a diffusion process.

Diffusion Convolutional LSTM replaces the matrix multiplication in a regular LSTM with diffusion convolution. It determines the future state of a certain cell in the graph by the inputs and past states of its local neighbors:

$$
\begin{bmatrix} i_t \\ f_t \\ o_t \end{bmatrix} = \sigma\big(W^{x} \star_g x_t + W^h \star_g h_{t-1} + W^c \circ c_{t-1} + b\big)
$$

where $W \star_g x = \sum_{i=1}^k \big(D^{-1}A\big)^i \cdot W \cdot x$ is the diffusion convolution.
