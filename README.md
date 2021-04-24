<a href="https://torchts.ai">
  <img width="350" src="./docs/source/_static/images/torchTS_logo.png" alt="TorchTS Logo" />
</a>

<hr/>

[![Conda](https://img.shields.io/conda/v/pytorch/torchts.svg)](https://anaconda.org/pytorch/torchts)
[![PyPI](https://img.shields.io/pypi/v/torchts.svg)](https://pypi.org/project/torchts)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Codecov](https://img.shields.io/codecov/c/github/pytorch/torchts.svg)](https://codecov.io/github/pytorch/torchts)

TorchTS is a Pytorch-based library for Time Series Data.

* Currently under active development!*

#### Why Time Series ?
Time series data modeling has broad significance in public health, finance and engineering. Traditional time series methods from statistics often rely on strong modeling assumptions, or are computationally expensive.  Given the rise of large-scale sensing data and significant advances in deep learning, the goal of the project is to develop an efficient and user-friendly deep learning library that would benefit the entire research community and beyond.

#### Why TorchTS ?
Existing time series analysis libraries include [statsmodels](https://www.statsmodels.org/stable/index.html),  [sktime](https://github.com/alan-turing-institute/sktime). However, these libraries only include traditional statistics tools such as ARMA or ARIMA, which do not have the state-of-the-art forecasting tools based on deep learning. [GluonTS](https://ts.gluon.ai/) is an open-source time series library developed by Amazon AWS, but is based on MXNet. [Pyro](https://pyro.ai/) is a probabilistic programming framework based on Pytorch, but is not focused on time series forecasting.

#### Benchmark


## Installation

**Installation Requirements**
- Python >= 3.7
- PyTorch >= 1.7
- scipy


##### Installing the latest release

The latest release of TorchTS is easily installed either via
[Anaconda](https://www.anaconda.com/distribution/#download-section) (recommended):
```bash
conda install torchts -c pytorch
```
or via `pip`:
```bash
pip install torchts
```

You can customize your PyTorch installation (i.e. CUDA version, CPU only option)
by following the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

***Important note for MacOS users:***
* Make sure your PyTorch build is linked against MKL (the non-optimized version
  of TorchTS can be up to an order of magnitude slower in some settings).
  Setting this up manually on MacOS can be tricky - to ensure this works properly,
  please follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
* If you need CUDA on MacOS, you will need to build PyTorch from source. Please
  consult the PyTorch installation instructions above.


## Getting Started

Here's a quick run down of the main components of a Bayesian optimization loop.
For more details see our [Documentation](https://torchts.ai/docs/introduction) and the
[Tutorials](https://torchts.ai/tutorials).

1. Multi-step forecasting
  ```python

  ```

2. Multivariate forecasting with side information
  ```python

  ```

3. Hybrid forecasting with PDEs
  ```python

  ```


## Citing TorchTS

If you use TorchTS, please cite the following paper:
> [TBD. TorchTS: A Framework for Efficient Time Series Modeling.](TBD)

```
@inproceedings{TBD,
  title={{TorchTS: A Framework for Efficient Time Series Modeling}},
  author={},
  booktitle = {TBD},
  year={TBD},
  url = {}
}
```

See [here](https://torchts.ai/docs/papers) for an incomplete selection of peer-reviewed papers that build off of TorchTS.


## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.


## License
TorchTS is MIT licensed, as found in the [LICENSE](LICENSE) file.
