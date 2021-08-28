<a href="https://rose-stl-lab.github.io/torchTS/">
  <img width="350" src="./docs/source/_static/images/torchTS_logo.png" alt="TorchTS Logo" />
</a>

---

[![Tests](https://github.com/Rose-STL-Lab/torchTS/workflows/Tests/badge.svg)](https://github.com/Rose-STL-Lab/torchTS/actions?query=workflow%3ATests)
[![Quality](https://github.com/Rose-STL-Lab/torchTS/workflows/Quality/badge.svg)](https://github.com/Rose-STL-Lab/torchTS/actions?query=workflow%3AQuality)
[![Docs](https://github.com/Rose-STL-Lab/torchTS/workflows/Docs/badge.svg)](https://github.com/Rose-STL-Lab/torchTS/actions?query=workflow%3ADocs)
[![Codecov](https://img.shields.io/codecov/c/github/Rose-STL-Lab/torchTS?label=Coverage&logo=codecov)](https://app.codecov.io/gh/Rose-STL-Lab/torchTS)
[![PyPI](https://img.shields.io/pypi/v/torchts?label=PyPI&logo=python)](https://pypi.org/project/torchts)
[![License](https://img.shields.io/github/license/Rose-STL-Lab/torchTS?label=License)](LICENSE)

TorchTS is a PyTorch-based library for time series data.

***Currently under active development!***

#### Why Time Series?

Time series data modeling has broad significance in public health, finance and engineering. Traditional time series methods from statistics often rely on strong modeling assumptions, or are computationally expensive. Given the rise of large-scale sensing data and significant advances in deep learning, the goal of the project is to develop an efficient and user-friendly deep learning library that would benefit the entire research community and beyond.

#### Why TorchTS?

Existing time series analysis libraries include [statsmodels](https://www.statsmodels.org/stable/index.html) and [sktime](https://github.com/alan-turing-institute/sktime). However, these libraries only include traditional statistics tools such as ARMA or ARIMA, which do not have the state-of-the-art forecasting tools based on deep learning. [GluonTS](https://ts.gluon.ai/) is an open-source time series library developed by Amazon AWS, but is based on MXNet. [Pyro](https://pyro.ai/) is a probabilistic programming framework based on PyTorch, but is not focused on time series forecasting.

## Installation

### Installation Requirements

TorchTS supports Python 3.7+ and has the following dependencies:

- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://pytorchlightning.ai/)
- [SciPy](https://www.scipy.org/)

### Installing the latest release

The latest release of TorchTS is easily installed either via `pip`:

```bash
pip install torchts
```

or via [conda](https://docs.conda.io/projects/conda/) from the [conda-forge](https://conda-forge.org/) channel (coming soon):

```bash
conda install torchts -c conda-forge
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

Check out our [documentation](https://rose-stl-lab.github.io/torchTS/) and
[tutorials](https://rose-stl-lab.github.io/torchTS/tutorials) (coming soon).

## Citing TorchTS

If you use TorchTS, please cite the following paper (coming soon):

> [TorchTS: A Framework for Efficient Time Series Modeling](TBD)

```bibtex
@inproceedings{TBD,
  title={{TorchTS: A Framework for Efficient Time Series Modeling}},
  author={TBD},
  booktitle = {TBD},
  year={TBD},
  url = {TBD}
}
```

See [here](https://rose-stl-lab.github.io/torchTS/papers) (coming soon) for a selection of peer-reviewed papers that either build off of TorchTS or were integrated into TorchTS.

## Contributing

Interested in contributing to TorchTS? Please see the [contributing guide](CONTRIBUTING.md) to learn how to help out.

## License

TorchTS is [MIT licensed](LICENSE).
