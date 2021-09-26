---
title: Autoregressive Model (AR)
slug: /autoregressive
---

Given a time series $x_1, \ldots, x_t$, a $p$-th order autoregressive model (denoted AR($p$)) is defined as a linear function of the input series $x$:

$$
x_t = \sum_{i=1}^p a_i x_{t-i} + e_t
$$

where $\{a_i\}$ are the model coefficients and the series $\{e_t\}$ can represent either a controlled external input or noise. Note that the expression

$$
\sum_{i=1}^p a_i x_{t-i} = a_1 x_{t-1} + a_2 x_{t-2} + \cdots + a_p x_{t-p}
$$

describes a convolution filter. We can implement AR($p$) using either a feedforward neural network with a rolling window or a convolutional network on the entire series.
