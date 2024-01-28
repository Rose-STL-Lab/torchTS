---
title: Quantile Regression
slug: /quantile-regression
---

Quantile regression uses the one-sided quantile loss to predict specific percentiles of the dependent variable. The quantile regression model uses the pinball loss function written as:

$$
L_{Quantile}\big(y,f(x),\theta,p\big) = min_\theta\{\mathbb{E}_{(x,y)\sim D}[(y - f(x))(p - \mathbb{1}\{y < f(x)\})]\}
$$

where $p$ is our fixed confidence interval parameterized by $\theta$. When the pinball loss is minimized, the result is the optimal quantile.

### Examples

---

```python
from torchts.nn.loss import quantile_loss
from torchts.nn.model import TimeSeriesModel

# initialize model
class Model(TimeSeriesModel):
    def __init__(self):
        ...

    def forward(self, x):
        ...

model = Model(
    ...,
    criterion=quantile_loss,
    criterion_args={"quantile": 0.05}
)

model.fit(...)
y_pred = model.predict(...)
```

[Full example](https://github.com/Rose-STL-Lab/torchTS/blob/main/examples/quantile-regression/lstm-quantile-regression.ipynb)
