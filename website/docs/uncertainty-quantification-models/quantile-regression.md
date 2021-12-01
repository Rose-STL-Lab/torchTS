---
title: Quantile Regression
slug: /quantile-regression
---

Quantile regression uses one-sided quantile loss to predict specific percentiles of the dependent variable. Our quantile regression model uses the pinball loss function written as:

$$L_{Quantile}(y,f(x),\theta,p))$$
$$=min_\theta\{\mathbb{E}_{(x,y)\sim D}[(y-f(x))(p-\mathbb{1}\{y< f(x)\})]\}$$

Where $p$ is our fixed confidence interval parameterized by $\theta$. When the pinball loss is minimized, the result is the optimal quantile.