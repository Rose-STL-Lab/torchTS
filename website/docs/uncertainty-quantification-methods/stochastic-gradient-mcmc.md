---
title: Stochastic Gradient MCMC (coming soon)
slug: /stochastic-gradient-mcmc
---

Stochastic Gradient MCMC (SG-MCMC) is a Bayesian uncertainty quantification method. This form of gradient descent is useful in allowing us to calculate our quantiles according to subsets of the training data set which are selected based on the posterior distribution over the parameter space. Also, we follow the stochastic gradient thermostat method (SGNHT), whose purpose is to control gradient noise, which keeps our distribution gaussian. We generate samples of model parameters $\theta$ as a function of our loss function L($\theta$), diffusion coefficients A, and learning rate h, in addition to auxiliary variables p $\epsilon$  $\mathbb{R}^d$ and $\zeta$ $\epsilon$  R. We randomly initialize $\theta$, p, and $\zeta$ and update according to the rule:

$$
\begin{aligned}
\theta_{k+1} &= \theta_k + p_kh \\
p_{k+1} &= p_k - \triangledown L(\theta)h - \zeta_kp_kh + \mathcal{N}(0,2Ah) \\
\zeta_{k+1} &= \zeta_k + \left(\frac{p^t_kp_k}{d} - 1\right)h
\end{aligned}
$$

Where after the kth iteration, $\theta$ follows the distribution of the posterior. By running for multiple $\theta$ with different samples according to the posterior, we quantify the uncertainty of our prediction.
