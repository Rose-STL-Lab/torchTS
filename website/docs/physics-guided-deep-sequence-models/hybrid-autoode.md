---
title: Hybrid AutoODE
slug: /hybrid-autoode
---

Assume the time series $x_t \in \mathbb{R}^d$ is governed by unknown differential equations and by other unknown factors that could affect its trajectory. The Hybrid AutoODE uses physics-guided models in conjunction with neural networks to improve the prediction of $x_t$. It is modelled by the following equations:

$$
\begin{aligned}
&\frac{dx}{dt} = f_\theta(t, x, u, F) \\
&\frac{du}{dt} = g_\theta(t, x, u, F) \\
&x(t_0) = x_0 \\
&u(t_0) = u_0
\end{aligned}
$$

where $u \in \mathbb{R}^p$ are the unobserved variables and $F$ is a neural network. The Hybrid AutoODE uses auto-differentiation to estimate the parameters $\theta$ of the equations and the neural network.
