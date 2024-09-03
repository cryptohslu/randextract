# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "RandExtract"
copyright = "2024, HSLU"
author = "Iyán Méndez Veiga"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_tabs.tabs",
    "sphinx_togglebutton",
]
source_suffix = {'.rst': 'restructuredtext'}
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "label"
todo_include_todos = True
todo_emit_warnings = False

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "use_sidenotes": True,
    "show_toc_level": 2,
    "show_navbar_depth": 1,
    "home_page_in_toc": True,
    "logo": {
        "image_light": "logo-light-mode.png",
        "image_dark": "logo-dark-mode.png",
        "alt-text": "Logo of randExtract",
    },
}

# -- Options for LaTeX -------------------------------------------------------
latex_engine = "xelatex"

# -- Options for autodoc -----------------------------------------------------
autoclass_content = "class"
autodoc_class_signature = "mixed"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
simplify_optional_unions = True
typehints_use_signature = False
typehints_use_signature_return = True
typehints_defaults = None
typehints_use_rtype = True
typehints_document_rtype = True
