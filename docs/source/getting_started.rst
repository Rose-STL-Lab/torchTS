Getting Started
===============

Make sure you have installed `torchTS`

In the following example, we will use the `torchTS` package to train a simple LSTM model on a time-series datasets. We will also enable uncertainty quantification so that we can get prediction intervals.

1. First, we will import necessary package.

.. code-block:: python

    import torch
    import torchts
    import numpy as np


2. Let's randomly generate a time-series dataset.

.. code-block:: python

    # generate linear time series data with some noise
    n = 200
    x_max = 10
    slope = 2
    scale = 2

    x = torch.from_numpy(np.linspace(-x_max, x_max, n).reshape(-1, 1).astype(np.float32))
    y = slope * x + np.random.normal(0, scale, n).reshape(-1, 1).astype(np.float32)

    plt.plot(x, y)
    plt.show()

We will get the following plots:

.. image:: ./_static/images/getting_started__dataset_plot.png
    :scale: 100%


3. Then, we can start selecting and training our model. In this example, we will use LSTM model.

.. code-block:: python

    model = LSTM(
        input_size,
        output_size,
        hidden_size,
        optimizer,
        interval=interval,
        optimizer_args=optimizer_args,
    )
    model.fit(x, y, max_epochs=max_epochs, batch_size=batch_size)


4. After model is trained, we can use it to predict the future values. And more importantly, since we enable uncertainty quantification method, we can also get a prediction interval!

.. code-block:: python

    y_preds = model.predict(x)


5. Let's plot prediction results

.. code-block:: python

    plt.plot(x, y, label="y_true")
    plt.plot(x, y_preds, label=["lower", "upper"])
    plt.legend()
    plt.show()


.. image:: ./_static/images/getting_started__pred_results_1.png
    :scale: 100%


Example prediction results for other datasets:
        

.. image:: ./_static/images/getting_started__sample_dataset.png
    :scale: 100%


.. image:: ./_static/images/getting_started__sample_results.png
    :scale: 100%
