---
title: MIS Regression
slug: /mis-regression
---

Mean interval score (MIS) regression directly minimizes MIS, a scoring function for predictions of confidence intervals. This is done by using MIS as the loss function for deep neural networks. The formula to calculate MIS is also known as Winkler loss and is expressed as:

$$
MIS = \frac{1}{h}\sum_{j=1}^{h}\left((u_{t+j} - l_{t+j}) + \frac{2}{\alpha}(l_{t+j} - y_{t+j})\mathbb{1}(y_{t+j} < l_{t+j}) + \frac{2}{\alpha}(y_{t+j} - u_{t+j})\mathbb{1}(y_{t+j} > u_{t+j})\right)
$$

where $u$ and $l$ are the upper and lower bounds respectively, and $\alpha$ is a fixed confidence level. Here $\alpha$ is equivalent to $1-CI$, where $CI$ is the desired confidence interval. Therefore, $\alpha=0.05$ for a $95\%$ confidence interval. There are 3 parts to this loss function, which summed together equal the total mean interval score.

1. Penalize distance between the upper and lower bounds.
2. Penalize distance between the lower bound and actual value by a ratio of $2/\alpha$ when the actual value is lower than the lower bound.
3. Penalize distance between the actual value and upper bound by a ratio of $2/\alpha$ when the actual value is higher than the upper bound.

Since the loss function for this model jointly includes the upper and lower bounds, the result outputs both, unlike with quantile regression.
