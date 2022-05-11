# Change Log

## [Unreleased]

### Added

- Added conda-forge README badge. ([#149](https://github.com/Rose-STL-Lab/torchTS/pull/149))
- Added pre-commit.ci config and README badge. ([#156](https://github.com/Rose-STL-Lab/torchTS/pull/156))
- Added quantile loss function for uncertainty quantification. ([#168](https://github.com/Rose-STL-Lab/torchTS/pull/168))
- Added input to base model constructor for loss function arguments. ([#168](https://github.com/Rose-STL-Lab/torchTS/pull/168))
- Added pytest-mock to dev dependencies. ([#168](https://github.com/Rose-STL-Lab/torchTS/pull/168))
- Added website descriptions for new features. ([#165](https://github.com/Rose-STL-Lab/torchTS/pull/165), [#169](https://github.com/Rose-STL-Lab/torchTS/pull/169))
- Added mean interval score loss function. ([#188](https://github.com/Rose-STL-Lab/torchTS/pull/188))
- Added API documentation to website. ([#206](https://github.com/Rose-STL-Lab/torchTS/pull/206), [#237](https://github.com/Rose-STL-Lab/torchTS/pull/237), [#238](https://github.com/Rose-STL-Lab/torchTS/pull/238))
- Added ODE solver and examples. ([#134](https://github.com/Rose-STL-Lab/torchTS/pull/134))

### Changed

- Updated documentation website. ([#125](https://github.com/Rose-STL-Lab/torchTS/pull/125))
- Replaced loop with list comprehension. ([#148](https://github.com/Rose-STL-Lab/torchTS/pull/148))
- Expanded automatic pull request labeling. ([#154](https://github.com/Rose-STL-Lab/torchTS/pull/154), [#204](https://github.com/Rose-STL-Lab/torchTS/pull/204))
- Expanded gitignore patterns. ([#155](https://github.com/Rose-STL-Lab/torchTS/pull/155))
- Updated flakehell pre-commit hook. ([#177](https://github.com/Rose-STL-Lab/torchTS/pull/177))
- Removed pull requests from security workflow runs. ([#185](https://github.com/Rose-STL-Lab/torchTS/pull/185))
- Switched from flakehell to flakeheaven. ([#203](https://github.com/Rose-STL-Lab/torchTS/pull/203))
- Removed pre-commit actions. ([#224](https://github.com/Rose-STL-Lab/torchTS/pull/224))

### Fixed

- Fixed equation parentheses in spatiotemporal documentation. ([#153](https://github.com/Rose-STL-Lab/torchTS/pull/153))

## [0.1.1] - 2021-08-31

This patch release sets dependency requirements for a `conda` installation. The original requirements were too strict for [conda-forge](https://conda-forge.org/).

### Added

- Added pre-commit to dev dependencies. ([#127](https://github.com/Rose-STL-Lab/torchTS/pull/127))

### Changed

- Changed CI workflows to run pre-commit with poetry. ([#131](https://github.com/Rose-STL-Lab/torchTS/pull/131))
- Moved common workflow steps to a composite action. ([#132](https://github.com/Rose-STL-Lab/torchTS/pull/132))
- Updated pre-commit hooks. ([#133](https://github.com/Rose-STL-Lab/torchTS/pull/133), [#135](https://github.com/Rose-STL-Lab/torchTS/pull/135))
- Relaxed dependency requirements. ([#139](https://github.com/Rose-STL-Lab/torchTS/pull/139))

### Fixed

- Fixed change log links. ([#126](https://github.com/Rose-STL-Lab/torchTS/pull/126), [#128](https://github.com/Rose-STL-Lab/torchTS/pull/128))
- Fixed contributing file link. ([#137](https://github.com/Rose-STL-Lab/torchTS/pull/137))
- Fixed Sphinx config metadata. ([#138](https://github.com/Rose-STL-Lab/torchTS/pull/138))

## [0.1.0] - 2021-08-16

Initial release

[unreleased]: https://github.com/Rose-STL-Lab/torchTS/compare/v0.1.1...main
[0.1.1]: https://github.com/Rose-STL-Lab/torchTS/releases/tag/v0.1.1
[0.1.0]: https://github.com/Rose-STL-Lab/torchTS/releases/tag/v0.1.0
