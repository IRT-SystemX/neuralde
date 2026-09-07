# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "neuralde"
copyright = "2026, IRT-SystemX, AirLiquide"
author = "martin.gonzalez, nelson luis pinto"
release = "1.1.0"

authors = [
("Martin Gonzalez", "martin.gonzalez@irt-systemx.fr"),
("Nelson Fernandez-Pinto", "nelson.fernandezpinto@airliquide.com"),
]
html_context = {
"authors": authors,
}

import os
import sys
sys.path.insert(0, os.path.abspath(f"{__file__}/../.."))
sys.path.insert(0,"..")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
]

extensions = [
    "sphinx.ext.githubpages","sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc", "sphinx.ext.napoleon",
              "sphinx_autodoc_typehints", "myst_parser", "nbsphinx"
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]