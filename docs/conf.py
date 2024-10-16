# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'scooby'
copyright = '2024, Hingerl, Martens et al.'
author = 'Hingerl, Martens et al.'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', 'autoapi.extension', 'nbsphinx']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

import sys
from pathlib import Path
sys.path.insert(0, str(Path('../').resolve()))

autoapi_dirs = ['../scooby']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_logo = "logo.png"
html_title = "scooby"
html_theme_options = {
    #"sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#003262",
        "color-brand-content": "#003262",
        "admonition-font-size": "var(--font-size-normal)",
        "admonition-title-font-size": "var(--font-size-normal)",
    },
    "dark_css_variables": {
        "color-brand-primary": "#e6f3ff",
        "color-brand-content": "#e6f3ff",
        "admonition-font-size": "var(--font-size-normal)",
        "admonition-title-font-size": "var(--font-size-normal)",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/gagneurlab/scooby",
            "html": "",
            "class": "fab fa-github",
        },
    ],
}


"""
html_static_path = ['_static']
html_sidebars = {
    '**': [
        'about.html',
        'searchfield.html',
        'navigation.html',
        'relations.html',
        'donate.html',
    ]
}
"""
