torchts.nn.loss
===============

Quatile Loss
------------

Quantile regression uses the one-sided quantile loss to predict specific percentiles of the dependent variable.
The quantile regression model uses the pinball loss function written as:

.. math::
    L_{Quantile}\big(y,f(x),\theta,p\big) = min_\theta\{\mathbb{E}_{(x,y)\sim D}[(y - f(x))(p - \mathbb{1}\{y < f(x)\})]\}

where :math:`p` is our fixed confidence interval parameterized by :math:`\theta`. When the pinball loss is minimized, the result is the optimal quantile.

.. autofunction:: torchts.nn.loss.quantile_loss
    :noindex:


Mean Interval Score Loss
------------------------

.. autofunction:: torchts.nn.loss.mis_loss
    :noindex:


Masked Mean Absolute Error Loss
--------------------------------

.. autofunction:: torchts.nn.loss.masked_mae_loss
    :noindex:

