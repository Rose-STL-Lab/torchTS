Installing torchTS
===================

Dependencies
^^^^^^^^^^^^
* Python 3.7+
* `PyTorch <https://pytorch.org/>`_
* `PyTorch Lightning <https://www.pytorchlightning.ai/>`_
* `SciPy <https://scipy.org/>`_


Installing the Latest Release
------------------------------

PyTorch Configuration
^^^^^^^^^^^^^^^^^^^^^
- Since torchTS is built upon PyTorch, you may want to customize your PyTorch configuration for your specific needs by following the `PyTorch installation instructions <https://pytorch.org/get-started/locally/>`_.

**Important note for MacOS users:**

- If you need CUDA on MacOS, you will need to build PyTorch from source. Please consult the PyTorch installation instructions above.

Typical Installation
^^^^^^^^^^^^^^^^^^^^

- To install torchTS through PyPI, execute this command::
    
    pip install torchts

Conda Installation
^^^^^^^^^^^^^^^^^^

- If you would like to install torchTS through conda, it is available through this command::

    conda install -c conda-forge torchts

torchTS Installation for Development Local Environment
------------------------------------------------------

- In order to develop torchTS, it is important to ensure you have the most up-to-date dependencies. `Poetry <https://python-poetry.org/>`_ is used by torchTS to help manage these dependencies in a elegant manner.

Clone repository
^^^^^^^^^^^^^^^^^^
- Begin by cloning the GitHub Repository::

    # Clone the latest version of torchTS from GitHub and navigate to the root directory
    git clone https://github.com/Rose-STL-Lab/torchTS.git
    cd torchTS


Use Poetry to Install Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Once you are in the root directory for torchTS, you can use the following command to install the most up-to-date dependencies for torchTS
- If you are unfamiliar with Poetry, follow the guides on `installation <https://python-poetry.org/docs/>`_ and `basic usage <https://python-poetry.org/docs/basic-usage/>`_ from the Poetry project’s documentation.::

    # install torchTS' dependencies through poetry
    poetry install

Running a simple notebook with your local environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Poetry essentially sets up a virtual environment that automatically configures itself with the dependencies needed to work with torchTS.
- Once you’ve installed the dependencies for torchTS through Poetry, we can run a Jupyter Notebook with a base kernel built upon torchTS’ using these commands::

    # Run this from the root directory of torchTS 
    poetry run jupyter notebook

- Similarly, we can run Python scripts through our compatible environment using this code configuration::

    # Run any python script through our new 
    poetry run [PYTHON FILE]

- Poetry is a very capable package management tool and we recommend you explore it’s functionalities further with `their documentation <https://python-poetry.org/docs/>`_ to get the most out of it.
