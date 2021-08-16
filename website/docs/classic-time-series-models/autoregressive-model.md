---
title: Autoregressive Model (AR)
slug: /autoregressive
---
Assume we are given a time series $x_1,\ldots,x_t$, a $P$-th order autoregressive model (denoted AR(P)) is defined as a linear function of the input series $x$:

$$
x_t = \sum_{i=1}^P a_i x_{t-i} + e_t
$$

where $\{a_i\}$ are the model coefficients. The series $\{e_t\}$ can represent either a controlled external input or noise. Note that:

$$
\sum_{i=1}^P a_i x_{t-i} = a_1x_{t-1} + a_2x_{t-2} + \cdots + a_Px_{t-P}
$$

describes a convolution filter. We can implement AR(P) using either a feedforward neural network with a rolling window or a convolutional network on the entire series.
