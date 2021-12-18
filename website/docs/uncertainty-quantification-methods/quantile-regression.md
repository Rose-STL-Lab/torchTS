---
title: Quantile Regression
slug: /quantile-regression
---

Quantile regression uses the one-sided quantile loss to predict specific percentiles of the dependent variable. The quantile regression model uses the pinball loss function written as:

$$
L_{Quantile}\big(y,f(x),\theta,p\big) = min_\theta\{\mathbb{E}_{(x,y)\sim D}[(y - f(x))(p - \mathbb{1}\{y < f(x)\})]\}
$$

where $p$ is our fixed confidence interval parameterized by $\theta$. When the pinball loss is minimized, the result is the optimal quantile.
