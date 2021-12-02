---
title: AutoODE
slug: /autoode
---

Assume the time series $x_t \in \mathbb{R}^d$ is governed by unknown differential equations:

$$
\begin{aligned}
&\frac{dx}{dt} = f_\theta(t, x, u) \\
&\frac{du}{dt} = g_\theta(t, x, u) \\
&x(t_0) = x_0 \\
&u(t_0) = u_0
\end{aligned}
$$

where $u \in \mathbb{R}^p$ are the unobserved variables. [AutoODE](https://arxiv.org/pdf/2011.10616.pdf) uses auto-differentiation to estimate the parameters $\theta$ of the equations.
