# Configuration file for the Sphinx documentation builder.
import datetime
# -- Project information
from jax_dna import __project__, __version__

project = __project__
author = "Ryan Krueger, Megan Engel, and the SSEC at JHU"
copyright = f"{datetime.date.today().year}, {author}"

release = __version__
version = __version__

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

autoapi_dirs = ["../../jax_dna"]
autoapi_type = "python"
autoapi_template_dir = "_templates/autoapi"
autodoc_typehints = "signature"
# if you want to debug uncomment this line
# autoapi_keep_files = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

# select theme from options below
html_theme = "sphinx_rtd_theme"
#html_theme = "sphinx_book_theme"
html_static_path = ["../_static"]
html_css_files = ["../_static/custom.css"]
html_logo = "../_static/SSEC_logo_vert_white_lg_1184x661.png"
html_title = f"{project} {release}"
html_theme_options = {}
if html_theme == "sphinx_book_theme":
    html_theme_options.update({
        "logo": {
            "image_light": "../_static/SSEC_logo_horiz_blue_1152x263.png",
            "image_dark": "../_staticautoapi/jax_dna/energy/configuration/index/SSEC_logo_vert_white_lg_1184x661.png",
            "text": f"{html_title}",
        },
        "repository_url": "https://github.com/rkruegs123/jax-dna-dev",
        "use_repository_button": True,
    })


# -- Options for EPUB output
epub_show_urls = "footnote"

import os
import sys
sys.path.insert(0, os.path.abspath("../jax_dna"))

def skip_irrelevant(app, what, name, obj, skip, options):
    if (
        "test" in name
    ):
        return True
    return skip

def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_irrelevant)