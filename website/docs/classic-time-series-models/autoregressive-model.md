---
title: Autoregressive Model (AR)
slug: /autoregressive
---
Assume we are given a time series $$x_1,\cdots, x_t$$, P-th order autoregressive (AR (P)) models the output as a linear function of the input series

$x_t = \sum_{i=1}^P a_i x_{t-i} + e_t$ 

where $\{a_i]}$ are the coefficients. The series $\{e_t\}$ can represent either a controlled external input or noise.

Note that the equation $\sum_{i=1}^p$ a_i x_{t-i} = a_1x_{t-1} + a_2x_{t-2}+\cdots + \cdots + a_Px_{t-P}$ describes a convolution filter. We can implement AR(P) using either a feedforward neural networks with rolling window or a  convolutional network on the entire series.
