---
title: AutoODE
slug: /autoode
---
Assume the time series $x_t\in \mathbb{R}^d$ is governed by unknown differential equations.
\begin{equation}
\label{def}
\begin{dcases}
&\frac{dx}{dt} = f_{\mathbf{\theta}}(t, x, u)\\
&\frac{du}{dt} = g_{\mathbf\theta}}(t, x, u)\\
& x_0 = {y}_0 \\
& x_0 = {u}_0.
\end{dcases}
\end{equation}
Where  $u \in \mathbb{R}^p$ are the unobserved variables.
[AutoODE](https://arxiv.org/pdf/2011.10616.pdf) uses auto-differentiation to estimate the parameters $\theta$ of the equations.
