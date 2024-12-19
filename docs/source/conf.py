# ruff: noqa
# mypy: ignore-errors

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from pathlib import Path

sys.path.insert(0, str(Path("../../src").absolute()))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "axtreme"
copyright = "2024, DNV AS. All rights reserved."
author = "Sebastian Winter, Kristoffer Skare"

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",  # upgrade to autodoc2if want to you myst markup in docstings
    "sphinx.ext.napoleon",
    "sphinx_argparse_cli",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinxcontrib.mermaid",
]

# Extenstion for myst_parser
myst_enable_extensions = [
    "dollarmath",
    "attrs_inline",
]

# The file extensions of source files.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = f"axtreme {release}"
html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/DNV_logo_RGB.jpg"
autodoc_default_options = {
    "member-order": "groupwise",
    "undoc-members": True,
    # "special-members": True,
    # TODO(sw 2024-12-5): using "inherited-members" might be a more elegant want to achieve the below.
    # "exclude-members": "__weakref__, __init__, __annotations__, __abstractmethods__, __module__, __parameters__, __subclasshook__",
    "exclude-members": "__weakref__",
}
autodoc_preserve_defaults = True

myst_heading_anchors = 3

todo_include_todos = True

# add markdown mermaid support
myst_fence_as_directive = ["mermaid"]
