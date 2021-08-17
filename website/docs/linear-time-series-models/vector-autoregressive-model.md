---
title: Vector Autoregressive Model (VAR)
slug: /vector
---
Given $k$ time series $x_1,\cdots, x_t$ with  $x_t \in \mathbb{R}^k$, a p-th order vector autoregressive (VAR) model generalizes the univariate AR model. It models the output as linear functions of the input series.

$x_t = \sum_{i=1}^p A_i x_{t-i} + e_t$

where $A_i \in \mathbb{R}^{k \times k}$ is a $k$ by $k$ matrix. The series $\{e_t\}$ can represent either a controlled external input or noise.
