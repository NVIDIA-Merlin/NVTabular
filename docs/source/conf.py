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
import subprocess
import sys

from natsort import natsorted

# rootdir = os.path.join(os.getenv("SPHINX_MULTIVERSION_SOURCEDIR", default="."), "..")
# sys.path.insert(0, rootdir)
sys.path.insert(0, os.path.abspath("../../"))


docs_dir = os.path.dirname(__file__)
repodir = os.path.abspath(os.path.join(__file__, r"../../.."))
gitdir = os.path.join(repodir, r".git")

# -- Project information -----------------------------------------------------

project = "NVTabular"
copyright = "2021, NVIDIA"  # pylint: disable=W0622
author = "NVIDIA"

# The full version, including alpha/beta/rc tags
release = "2021"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx_external_toc",
    "sphinx_multiversion",
    "sphinx_rtd_theme",
    "sphinx_markdown_tables",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.copydirs",
]

# MyST configuration settings
external_toc_path = "toc.yaml"
myst_enable_extensions = [
    "deflist",
    "html_image",
    "linkify",
    "replacements",
    "tasklist",
]
myst_linkify_fuzzy_links = False
myst_heading_anchors = 3
jupyter_execute_notebooks = "off"

# The API documents are RST and include `.. toctree::` directives.
suppress_warnings = ["etoc.toctree", "myst.header", "misc.highlighting_failure"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "examples/tensorflow/*",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
}
html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

source_suffix = [".rst", ".md"]

# Whitelist pattern for tags (set to None to ignore all tags)
# Determine if Sphinx is reading conf.py from the checked out
# repo (a Git repo) vs SMV reading conf.py from an archive of the repo
# at a commit (not a Git repo).
if os.path.exists(gitdir):
    tag_refs = subprocess.check_output(["git", "tag", "-l", "v*"]).decode("utf-8").split()
    tag_refs = natsorted(tag_refs)[-6:]
    smv_tag_whitelist = r"^(" + r"|".join(tag_refs) + r")$"
else:
    # SMV is reading conf.py from a Git archive of the repo at a specific commit.
    smv_tag_whitelist = r"^v.*$"

# Only include main branch for now
smv_branch_whitelist = "^main$"

smv_refs_override_suffix = "-docs"

html_sidebars = {"**": ["versions.html"]}

autodoc_inherit_docstrings = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
    "member-order": "bysource",
}

autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "cudf": ("https://docs.rapids.ai/api/cudf/stable/", None),
    "distributed": ("https://distributed.dask.org/en/latest/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "merlin-core": ("https://nvidia-merlin.github.io/core/main", None),
    "merlin-systems": ("https://nvidia-merlin.github.io/systems/main", None),
}

copydirs_additional_dirs = [
    "../../examples/",
]
copydirs_file_rename = {
    "README.md": "index.md",
}
