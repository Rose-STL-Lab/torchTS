---
title: MIS Regression
slug: /mis-regression
---

Mean interval score (MIS) regression directly minimizes MIS, a scoring function for predictions of intervals, to estimate confidence intervals. This is done by using MIS as the loss function for deep neural networks. The formula to calculate MIS is also known as Winkler loss and can be written as:

$MIS = \frac{1}{h}\sum_{j=1}^{h}((u_{t+j}-l_{t+j})+\frac{2}{\alpha}(l_{t+j}-y_{t+j})\mathbbm{1}(y_{t+j}<l_{t+j})+\frac{2}{\alpha}(y_{t+j}-u_{t+j})\mathbbm{1}(y_{t+j}>u_{t+j}))$

There are 3 parts to this loss function, which summed together equal the total mean interval score.
1. Penalize distance between the upper and lower bounds
2. Penalize by a ratio of 2/α when the actual value is lower than the lower bound
3. Penalize by a ratio of 2/α when the actual value is higher than the predicted value

Because the loss function for this model jointly includes the upper and lower bounds, the result outputs both, unlike with quantile regression.