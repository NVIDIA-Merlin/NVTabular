# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import sphinx
import sphinx.domains
from recommonmark.parser import CommonMarkParser

rootdir = os.path.join(os.getenv("SPHINX_MULTIVERSION_SOURCEDIR", default="."), "..")
sys.path.insert(0, rootdir)


# -- Project information -----------------------------------------------------

project = "NVTabular"
copyright = "2020, NVIDIA"
author = "NVIDIA"

# The full version, including alpha/beta/rc tags
release = "2020"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_multiversion",
    "sphinx_rtd_theme",
    "recommonmark",
    "sphinx_markdown_tables",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

source_parsers = {".md": CommonMarkParser}
source_suffix = [".rst", ".md"]

nbsphinx_allow_errors = True
html_show_sourcelink = False

# Whitelist pattern for tags (set to None to ignore all tags)
smv_tag_whitelist = r"^v.*$"
# Only include main branch for now
smv_branch_whitelist = "^main$"

html_sidebars = {"**": ["versions.html"]}

# certain references in the README couldn't be autoresolved here,
# hack by forcing to the either the correct documentation page (examples)
# or to a blob on the repo
_REPO = "https://github.com/NVIDIA/NVTabular/blob/main/"
_URL_MAP = {
    "./examples": "examples/index",
    "examples/rossmann/": "examples/rossmann/index",
    "examples/criteo-example.ipynb": "examples/criteo",
    "./CONTRIBUTING": _REPO + "/CONTRIBUTING.md",
    "./Operators": _REPO + "/Operators.md",
}


class GitHubDomain(sphinx.domains.Domain):
    def resolve_any_xref(self, env, docname, builder, target, node, contnode):
        resolved = _URL_MAP.get(target)
        if resolved:
            contnode["refuri"] = resolved
            return [("github:any", contnode)]
        return []


def setup(app):
    app.add_domain(GitHubDomain)
