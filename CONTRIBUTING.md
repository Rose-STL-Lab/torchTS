# Contributing to TorchTS

:wave: Welcome to TorchTS! Thank you for showing interest and taking the time to contribute. The TorchTS team would love to have your contribution.

The following is a set of guidelines for contributing to TorchTS on GitHub. Please read carefully, but note that these are mostly guidelines, not rules. Use your best judgement, and feel free to propose changes to this document in a pull request.

Table of contents:

- [Issues](#issues)
  - [Reporting Bugs](#reporting-bugs)
    - [Before submitting a bug report](#before-submitting-a-bug-report)
    - [How do I submit a bug report?](#how-do-i-submit-a-bug-report)
  - [Suggesting Enhancements](#suggesting-enhancements)
    - [Before submitting an enhancement suggestion](#before-submitting-an-enhancement-suggestion)
    - [How do I submit an enhancement suggestion?](#how-do-i-submit-an-enhancement-suggestion)
- [Pull Requests](#pull-requests)
  - [Contributing to Code](#contributing-to-code)
    - [Picking an issue](#picking-an-issue)
    - [Local development](#local-development)
    - [Running tests](#running-tests)
  - [Contributing to Documentation](#contributing-to-documentation)
  - [Creating Pull Requests](#creating-pull-requests)
- [License](#license)

## Issues

### Reporting Bugs

This section guides you through submitting a bug report for TorchTS. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

#### Before submitting a bug report

Before submitting bug reports, please search the existing issues on the [issue tracker](https://github.com/Rose-STL-Lab/torchTS/issues) to verify your issue has not already been submitted. Issues pertaining to bugs are usually marked with the [bug](https://github.com/Rose-STL-Lab/torchTS/issues?q=is%3Aissue+label%3Abug) label.

> **Note:** If you find a **Closed** issue that seems to be the same thing you are experiencing, open a new issue and include a link to the original issue in the body of your new one.

#### How do I submit a bug report?

Bugs are tracked on the [issue tracker](https://github.com/Rose-STL-Lab/torchTS/issues) where you can create a new one. When creating a bug report, please include as many details as possible. Explain the problem and include additional details to help maintainers reproduce the problem:

- **Use a clear and descriptive title** for the issue to identify the problem.
- **Describe the exact steps which reproduce the problem** in as many details as possible.
- **Provide specific examples to demonstrate the steps to reproduce the issue**. Include links to files or GitHub projects, or copy-paste-able snippets, which you use in those examples.
- **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
- **Explain which behavior you expected to see instead and why.**

Provide additional context by answering these questions:

- **Did the problem start happening recently** (e.g. after updating to a new version of TorchTS) or has this always been a problem?
- If the problem started happening recently, **can you reproduce the problem in an older version of TorchTS?** What is the most recent version in which the problem does not happen?
- **Can you reliably reproduce the issue?** If not, provide details about how often the problem happens and under what conditions it normally happens.

Include details about your configuration and environment:

- **Which version of TorchTS are you using?**
- **Which version of Python are you using?**
- **Which OS type and version are you using?**

### Suggesting Enhancements

This section guides you through submitting a suggestion for enhancements to TorchTS, including completely new features and minor improvements to existing functionality. Following these guidelines helps maintainers and the community understand your suggestion and find related suggestions.

#### Before submitting an enhancement suggestion

Before submitting enhancement suggestions, please search the existing issues on the [issue tracker](https://github.com/Rose-STL-Lab/torchTS/issues) to verify your issue has not already been submitted. Issues pertaining to enhancements are usually marked with the [enhancement](https://github.com/Rose-STL-Lab/torchTS/issues?q=is%3Aissue+label%3Aenhancement) label.

> **Note:** If you find a **Closed** issue that seems to be the same thing you are suggesting, please read the conversation to learn why it was not incorporated.

#### How do I submit an enhancement suggestion?

Enhancement suggestions are tracked on the [issue tracker](https://github.com/Rose-STL-Lab/torchTS/issues) where you can create a new one. When creating an enhancement suggestion, please include as many details as possible and provide the following information:

- **Use a clear and descriptive title** for the issue to identify the suggestion.
- **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
- **Provide specific examples to demonstrate the steps**.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why.

## Pull Requests

### Contributing to Code

#### Picking an issue

If you would like to take on an open issue, feel free to comment on it in the [issue tracker](https://github.com/Rose-STL-Lab/torchTS/issues). We are more than happy to discuss solutions to open issues. If you are particularly adventurous, consider addressing an issue labeled [help wanted](https://github.com/Rose-STL-Lab/torchTS/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).

> **Note:** If you are a first time contributor and are looking for an issue to take on, you might want to look through the issues labeled [good first issue](https://github.com/Rose-STL-Lab/torchTS/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

#### Local development

You will need the TorchTS source to start contributing to the codebase. Refer to the [documentation](https://rose-stl-lab.github.io/torchTS/docs/) to start using TorchTS. You will first need to clone the repository using `git` and place yourself in the new local directory:

```bash
git clone git@github.com:Rose-STL-Lab/torchTS.git
cd torchTS
```

> **Note:** We recommend that you use a personal [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) for this step. If you are new to GitHub collaboration, you can refer to the [Forking Projects Guide](https://guides.github.com/activities/forking/).

TorchTS uses [Poetry](https://python-poetry.org/) for dependency management and we recommend you do the same when contributing to TorchTS. Refer to the [installation instructions](https://python-poetry.org/docs/#installation) to determine how to install Poetry on your system. Once installed, you can create a virtual environment and install TorchTS including all dependencies with the `install` command:

```bash
poetry install
```

> **Note:** Poetry uses the active Python installation to create the virtual environment. You can determine your current Python version with `python --version` and find the location of your Python executable with `which python`.

#### Running tests

Once your changes are complete, make sure that the tests pass on your machine:

```bash
poetry run pytest tests/
```

> **Note:** Your code must always be accompanied by corresponding tests. Your pull request **will not be merged** if tests are not present.

TorchTS uses [pre-commit](https://pre-commit.com/) to run a series of checks that ensure all files adhere to a consistent code style and satisfy desired coding standards. These include [black](https://github.com/psf/black) and [pyupgrade](https://github.com/asottile/pyupgrade) to format code, [isort](https://github.com/PyCQA/isort) to sort import statements, and [Flake8](https://github.com/PyCQA/flake8) to check for common coding errors.

To make sure that you do not accidentally commit code that does not follow the coding style, you can run these checks with the following command:

```bash
poetry run pre-commit run --all-files
```

> **Note:** Many of the pre-commit hooks modify your code if necessary. If pre-commit fails, it will oftentimes pass when run a second time.

Failure to satisfy these checks will cause the CI to fail and your pull request **will not be merged**.

### Contributing to Documentation

One of the simplest ways to get started contributing to a project is through improving documentation. You can help by adding missing sections, editing the existing content so it is more accessible, or creating new content (tutorials, FAQs, etc).

Issues pertaining to the documentation are usually marked with the [documentation](https://github.com/Rose-STL-Lab/torchTS/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation) label.

### Creating Pull Requests

- Be sure that your pull request contains tests that cover the changed or added code.
- If your changes warrant a documentation change, the pull request must also update the documentation.

> **Note:** Make sure your branch is [rebased](https://docs.github.com/en/get-started/using-git/about-git-rebase) against the latest `main` branch. A maintainer might ask you to ensure the branch is up-to-date prior to merging your pull request if changes have conflicts.

All pull requests, unless otherwise instructed, need to be first accepted into the `main` branch.

## License

By contributing to TorchTS, you agree that your contributions will be licensed under the [LICENSE](LICENSE) file in the root directory of this source tree.
