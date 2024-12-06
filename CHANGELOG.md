# Changelog

All notable changes to the [my-package] project will be documented in this file.<br>
The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Solved
* Sphinx documentation: Resolved issue that documentation of class members was generated twice.

### Added
* Sphinx documentation: Added extension to support Markdown-based diagrams created with Mermaid.


## [0.1.9] - 2024-09-27

### Removed
* removed stub files for `torch` from ./stubs folder.  I left them by mistake. They were not meant to be part of the template (and were anyway work in progress).


## [0.1.8] - 2024-09-24

### Changed
* merged latest learnings gained in https://github.com/dnv-innersource/mvf repository, aiming to have `python_project_template` represent latest best practice.


## [0.1.7] - 2024-09-19

### Changed
* Changed from `pip`/`tox` to `uv` as package manager
* README.md : Completely rewrote section "Development Setup", introducing `uv` as package manager.
* Changed publishing workflow to use OpenID Connect (Trusted Publisher Management) when publishing to PyPI
* Updated copyright statement
* VS Code settings: Turned off automatic venv activation
* Replaced black formatter with ruff formatter

### Added
* Added `mypy` as static type checker (in addition to `pyright`)

### Removed
* VS Code settings: Removed the setting which added the /src folder to PythonPath. This is no longer necessary. `uv` installs the project itself as a package in "editable" mode, which removes the need to manually add /src to the PythonPath environment variable.

### GitHub workflows
* (all workflows): Adapted to use `uv` as package manager
* _test_future.yml : updated Python version to 3.13.0-alpha - 3.13.0
* _test_future.yml : updated name of test job to 'test313'

### Dependencies
* updated to ruff>=0.6.3  (from ruff==0.4.2)
* updated to pyright>=1.1.378  (from pyright==1.1.360)
* updated to setup-python@v5  (from setup-python@v4)
* updated to actions-gh-pages@v4  (from actions-gh-pages@v3)
* updated to upload-artifact@v4  (from upload-artifact@v3)
* updated to download-artifact@v4  (from download-artifact@v3)
* updated to sourcery>=1.22  (from sourcery==1.16)
* updated to dictIO>=0.3.4  (from dictIO>=0.3.1)


## [0.1.6] - 2024-02-20

### Changed
* Moved all project configuration from setup.cfg to pyproject.toml
* Moved all tox configuration from setup.cfg to tox.ini.
* Moved pytest configuration from pyproject.toml to pytest.ini
* Deleted setup.cfg

### Dependencies
* updated to black[jupyter]==24.1  (from black[jupyter]==23.12)
* updated to ruff==0.2.1  (from ruff==0.1.8)
* updated to pyright==1.1.350  (from pyright==1.1.338)
* updated to sourcery==1.15  (from sourcery==1.14)
* updated to dictIO>=0.3.1  (from dictIO>=0.2.9)


## [0.1.5] - 2023-11-08

### Changed

* incorporated latest updates introduced in mvx


## [0.1.4] - 2023-09-25

### Dependencies

* Updated dependencies to latest versions


## [0.1.3] - 2023-08-24

### Changed

* GitHub workflow publish_release.yml: corrected smaller errors
* Explicitly removed .env file from remote repository
* Updated README.md to include guidance on how to create a .env file locally
* dependencies: updated packages in requirements-dev.txt to latest versions


## [0.1.2] - 2023-06-22

### Changed

* Modularized GitHub workflows
* requirements-dev.txt: Updated dependencies to latest versions
* setup.cfg: indicated supported Python versions as py310 and py311 <br>
  (from formerly py39 and py310)
* GitHub workflows: changed default Python version from 3.10 to 3.11


## [0.1.1] - 2023-05-02

### Changed

* requirements-dev.txt: Updated dependencies to latest versions


## [0.1.0] - 2023-02-21

### Changed

* pyproject.toml: Changed ruff configuration to by default allow Uppercase variable names in functions. <br>
(As this is a very common case in science calculus)
* README.md: Changed install infos for development setup to pip install requirements-dev.txt (not requirements.txt)


## [0.0.1] - 2023-02-21

* Initial release

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
[unreleased]: https://github.com/dnv-innersource/my-package/compare/v0.1.9...HEAD
[0.1.9]: https://github.com/dnv-innersource/my-package/releases/tag/v0.1.8...v0.1.9
[0.1.8]: https://github.com/dnv-innersource/my-package/releases/tag/v0.1.7...v0.1.8
[0.1.7]: https://github.com/dnv-innersource/my-package/releases/tag/v0.1.6...v0.1.7
[0.1.6]: https://github.com/dnv-innersource/my-package/releases/tag/v0.1.5...v0.1.6
[0.1.5]: https://github.com/dnv-innersource/my-package/releases/tag/v0.1.4...v0.1.5
[0.1.4]: https://github.com/dnv-innersource/my-package/releases/tag/v0.1.3...v0.1.4
[0.1.3]: https://github.com/dnv-innersource/my-package/releases/tag/v0.1.2...v0.1.3
[0.1.2]: https://github.com/dnv-innersource/my-package/releases/tag/v0.1.1...v0.1.2
[0.1.1]: https://github.com/dnv-innersource/my-package/releases/tag/v0.1.0...v0.1.1
[0.1.0]: https://github.com/dnv-innersource/my-package/releases/tag/v0.0.1...v0.1.0
[0.0.1]: https://github.com/dnv-innersource/my-package/releases/tag/v0.0.1
[my-package]: https://github.com/dnv-innersource/my-package
