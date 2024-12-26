# Changelog

All notable changes to the [axtreme] project will be documented in this file.<br>
The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.1.1] - 2024-12-27
### Added:
Major addition of new qoi method `MarginalCDFExtrapolation`. The following helpers are alse added.
* `utils/numerical_precision`: quantify the precision possible with different datatypes
* `distributions/mixture`: The contains mixture distributions required for marginalisation
* `distributions/icdf`: optimisation methods for finding icdfs where not availab.
* Associated test for these pieces of functionality.

### Changed
* tests/qoi/test_gp_brute_force_system.py : Improved type-checking
* pyproject.toml:
  * cleaned up and reorganized dependencies
  * removed orphaned dependency `pytest-django`
* .pre-commit-config.yaml : updated with latest changes in python_project_template
* README.md : updated with latest changes from python_project_template
* GitHub workflows _test.yml and _test_future.yml : rewrote how pytest gets called in a cleaner way
* `eval/object_logging: unpack_object`: minor update to support `type` attributes.
* `eval/qoi_helpers.py`: minor updates to prevent divide by 0 warnings
* `tutorials/end_to_end_v2.py` -> `tutorials/basic_example.py`. Updated to use `MarginalCDFExtrapolation`.
* `data/importance_dataset.py`: Updated typing and used `torch`'s `StackedDataset`.

### Fixed
* Minor updates to docs.


### Dependencies
* Updated to pyarrow>=18.1  (from pyarrow>=17.0)
* Updated to statsmodels>=0.14.4  (from statsmodels>=0.14.2)


## [0.1.0] - 2024-12-10

* Initial release.


## [0.0.1] - 2023-02-21

### Added

* added this

### Changed

* changed that

### Dependencies

* updated to some_package_on_pypi>=0.1.0

### Fixed

* fixed issue #12345

### Deprecated

* following features will soon be removed and have been marked as deprecated:
    * function x in module z

### Removed

* following features have been removed:
    * function y in module z


<!-- Markdown link & img dfn's -->
[unreleased]: https://github.com/dnv-opensource/axtreme/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/dnv-opensource/axtreme/releases/tag/v0.1.1
[0.1.0]: https://github.com/dnv-opensource/axtreme/releases/tag/v0.1.0
[0.0.1]: https://github.com/dnv-opensource/axtreme/releases/tag/v0.0.1
[axtreme]: https://github.com/dnv-opensource/axtreme
