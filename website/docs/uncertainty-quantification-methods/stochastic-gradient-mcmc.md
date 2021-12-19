---
title: Stochastic Gradient MCMC (coming soon)
slug: /stochastic-gradient-mcmc
---

Stochastic gradient Markov chain Monte Carlo (SG-MCMC) is a Bayesian uncertainty quantification method. This form of gradient descent is useful in calculating quantiles according to subsets of the training data, which are selected based on the posterior distribution over the parameter space. This implementation also follows the stochastic gradient thermostat method (SGNHT), whose purpose is to control gradient noise, which keeps the distribution Gaussian. We generate samples of model parameters $\theta$ as a function of the loss function $L(\theta)$, diffusion coefficients $A$, and learning rate $h$, in addition to auxiliary variables $p \in \mathbb{R}^d$ and $\zeta \in \mathbb{R}^d$. The values of $\theta$, $p$, and $\zeta$ are randomly initialized and updated according to the rule:

$$
\begin{aligned}
\theta_{k+1} &= \theta_k + p_kh \\
p_{k+1} &= p_k - \triangledown L(\theta)h - \zeta_kp_kh + \mathcal{N}(0,2Ah) \\
\zeta_{k+1} &= \zeta_k + \left(\frac{p^t_kp_k}{d} - 1\right)h
\end{aligned}
$$

where after the $k$th iteration, $\theta$ follows the distribution of the posterior. We quantify the uncertainty of our prediction by running for multiple $\theta$ with different samples according to the posterior.
