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
import re
import sys
import warnings

from docutils import nodes
from recommonmark.parser import CommonMarkParser
from sphinx.errors import SphinxError

rootdir = os.path.join(os.getenv("SPHINX_MULTIVERSION_SOURCEDIR", default="."), "..")
sys.path.insert(0, rootdir)
docs_dir = os.path.dirname(__file__)


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
    "sphinx.ext.intersphinx",
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
html_static_path = []

source_parsers = {".md": CommonMarkParser}
source_suffix = [".rst", ".md"]

nbsphinx_allow_errors = True
html_show_sourcelink = False

# Whitelist pattern for tags (set to None to ignore all tags)
smv_tag_whitelist = r"^v.*$"
# Only include main branch for now
smv_branch_whitelist = "^main$"

html_sidebars = {"**": ["versions.html"]}


def setup(app):
    # Fixing how references (local links) work with Markdown
    app.connect("doctree-read", collect_ref_data)
    app.connect("doctree-resolved", process_refs)


# Adapted from
# https://github.com/apiaryio/dredd/blob/a0e999d69ee840778de191fd03acbdeda86e27e2/docs/conf.py#L172:L285  # noqa
# (released under the MIT license)
# -- Markdown References --------------------------------------------------


def collect_ref_data(app, doctree):
    """
    Finds all anchors and references (local links) within documents,
    and saves them as meta data
    """
    filename = doctree.attributes["source"]

    # this needs to happen to make this work with sphinx-multiversion
    metadata = app.config.smv_metadata or {}
    current_version = app.config.smv_current_version
    if metadata and current_version:
        sourcedir = metadata.get(current_version, {}).get("sourcedir")
        if sourcedir and filename.startswith(sourcedir):
            filename = filename[len(sourcedir) :]

    # otherwise lets just split off the current directory (not sphinx multiversion)
    filename = filename.replace(docs_dir, "").lstrip("/")
    docname = filename.replace(".md", "")

    anchors = []
    references = []

    for node in doctree.traverse(nodes.raw):
        if "name=" in node.rawsource:
            match = re.search(r'name="([^\"]+)', node.rawsource)
            if match:
                anchors.append(match.group(1))
        elif "id=" in node.rawsource:
            match = re.search(r'id="([^\"]+)', node.rawsource)
            if match:
                anchors.append(match.group(1))

    for node in doctree.traverse(nodes.section):
        for target in frozenset(node.attributes.get("ids", [])):
            anchors.append(target)

    for node in doctree.traverse(nodes.reference):
        uri = node.get("refuri")
        if uri and not uri.startswith(("http://", "https://")):
            ref = to_reference(uri, basedoc=docname)
            references.append(ref)

    app.env.metadata[docname]["anchors"] = anchors
    app.env.metadata[docname]["references"] = references


def process_refs(app, doctree, docname):
    """
    Fixes all references (local links) within documents, breaks the build
    if it finds any links to non-existent documents or anchors.
    """
    references = app.env.metadata[docname].get("references", [])
    if not references:
        return

    for reference in references:
        referenced_docname, anchor = parse_reference(reference)

        if referenced_docname not in app.env.metadata:
            message = "Document '{}' is referenced from '{}', but it could not be found"
            warnings.warn(message.format(referenced_docname, docname))
            continue

        if anchor and anchor not in app.env.metadata[referenced_docname]["anchors"]:
            message = "Section '{}#{}' is referenced from '{}', but it could not be found"
            raise SphinxError(message.format(referenced_docname, anchor, docname))
        for node in doctree.traverse(nodes.reference):
            uri = node.get("refuri")
            if to_reference(uri, basedoc=docname) == reference:
                node["refuri"] = to_uri(app, referenced_docname, anchor)


def to_uri(app, docname, anchor=None):
    uri = ""
    smv_current_version = app.config.smv_current_version
    if smv_current_version:
        uri = "/NVTabular/{}".format(smv_current_version)

    uri += "/{}.html".format(docname)
    if anchor:
        uri += "#{}".format(anchor)

    return uri


def to_reference(uri, basedoc=None):
    """
    Helper function, compiles a 'reference' from given URI and base
    document name
    """
    if "#" in uri:
        filename, anchor = uri.split("#", 1)
        filename = filename or basedoc
    else:
        filename = uri or basedoc
        anchor = None

    if not filename:
        message = "For self references like '{}' you need to provide the 'basedoc' argument".format(
            uri
        )
        raise ValueError(message)

    prefixes = ["docs/source/", "./docs/source/"]
    for prefix in prefixes:
        if filename.startswith(prefix):
            filename = filename[len(prefix) :]

    reference = os.path.splitext(filename.lstrip("/"))[0]
    if anchor:
        reference += "#" + anchor

    return reference


def parse_reference(reference):
    """
    Helper function, parses a 'reference' to document name and anchor
    """
    if "#" in reference:
        docname, anchor = reference.split("#", 1)
    else:
        docname = reference
        anchor = None
    return docname, anchor


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "cudf": ("https://docs.rapids.ai/api/cudf/stable/", None),
    "distributed": ("https://distributed.dask.org/en/latest/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

autodoc_inherit_docstrings = False
