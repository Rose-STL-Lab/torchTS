<a href="https://rose-stl-lab.github.io/torchTS/">
  <img width="350" src="./docs/source/_static/images/torchTS_logo.png" alt="TorchTS Logo" />
</a>

---

[![Tests](https://github.com/Rose-STL-Lab/torchTS/workflows/Tests/badge.svg)](https://github.com/Rose-STL-Lab/torchTS/actions?query=workflow%3ATests)
[![Quality](https://github.com/Rose-STL-Lab/torchTS/workflows/Quality/badge.svg)](https://github.com/Rose-STL-Lab/torchTS/actions?query=workflow%3AQuality)
[![Docs](https://github.com/Rose-STL-Lab/torchTS/workflows/Docs/badge.svg)](https://github.com/Rose-STL-Lab/torchTS/actions?query=workflow%3ADocs)
[![Codecov](https://img.shields.io/codecov/c/github/Rose-STL-Lab/torchTS?label=Coverage&logo=codecov)](https://app.codecov.io/gh/Rose-STL-Lab/torchTS)
[![Conda](https://img.shields.io/conda/v/pytorch/torchts?label=Conda&logo=anaconda)](https://anaconda.org/pytorch/torchts)
[![PyPI](https://img.shields.io/pypi/v/torchts?label=PyPI&logo=python)](https://pypi.org/project/torchts)
[![License](https://img.shields.io/github/license/Rose-STL-Lab/torchTS?label=License)](LICENSE)

TorchTS is a PyTorch-based library for time series data.

***Currently under active development!***

#### Why Time Series?

Time series data modeling has broad significance in public health, finance and engineering. Traditional time series methods from statistics often rely on strong modeling assumptions, or are computationally expensive.  Given the rise of large-scale sensing data and significant advances in deep learning, the goal of the project is to develop an efficient and user-friendly deep learning library that would benefit the entire research community and beyond.

#### Why TorchTS?

Existing time series analysis libraries include [statsmodels](https://www.statsmodels.org/stable/index.html),  [sktime](https://github.com/alan-turing-institute/sktime). However, these libraries only include traditional statistics tools such as ARMA or ARIMA, which do not have the state-of-the-art forecasting tools based on deep learning. [GluonTS](https://ts.gluon.ai/) is an open-source time series library developed by Amazon AWS, but is based on MXNet. [Pyro](https://pyro.ai/) is a probabilistic programming framework based on PyTorch, but is not focused on time series forecasting.

#### Benchmark

## Installation

### Installation Requirements

- Python >= 3.7
- PyTorch >= 1.7
- scipy

### Installing the latest release

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

- Make sure your PyTorch build is linked against MKL (the non-optimized version
  of TorchTS can be up to an order of magnitude slower in some settings).
  Setting this up manually on MacOS can be tricky - to ensure this works properly,
  please follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
- If you need CUDA on MacOS, you will need to build PyTorch from source. Please
  consult the PyTorch installation instructions above.

## Getting Started

Here's a quick run down of the main components of a deep learning forecaster.
For more details see our [Documentation](https://rose-stl-lab.github.io/torchTS/) and the
[Tutorials](https://torchts.ai/tutorials).

1. Multi-step forecasting

  ```python

  ```

2. Spatiotemporal forecasting 
  ```python

  ```

3. Hybrid forecasting with PDEs

  ```python

  ```

## Citing TorchTS

If you use TorchTS, please cite the following paper:
> [TBD. TorchTS: A Framework for Efficient Time Series Modeling.](TBD)

```bibtex
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
