# Configuration file for the Sphinx documentation builder.

# -- Project information
from jax_dna import __project__, __version__


project = __project__
copyright = '2024, Ryan Krueger'
author = 'Ryan Krueger'

release = __version__
version = __version__

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

autosummary_generate = True

# ignore test modules
exclude_patterns = ['**/tests/*.py']


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

# select theme from options below
html_theme = 'sphinx_rtd_theme'
#html_theme = 'sphinx_book_theme'
html_static_path = ['../_static']
html_css_files = ['../_static/custom.css']
html_logo = '../_static/SSEC_logo_vert_white_lg_1184x661.png'
html_title = f'{project} {release}'
html_theme_options = {}
if html_theme == 'sphinx_book_theme':
    html_theme_options.update({
        'logo': {
            'image_light': '../_static/SSEC_logo_horiz_blue_1152x263.png',
            'image_dark': '../_static/SSEC_logo_vert_white_lg_1184x661.png',
            'text': f'{html_title}',
        },
        'repository_url': 'https://github.com/rkruegs123/jax-dna-dev',
        'use_repository_button': True,
    })


# -- Options for EPUB output
epub_show_urls = 'footnote'

import os
import sys
sys.path.insert(0, os.path.abspath('../jax_dna'))

def skip_tests(app, what, name, obj, skip, options):
    # Skip any module or member that is in a 'tests' directory
    if hasattr(obj, '__module__') and 'tests' in obj.__module__.split('.'):
        return True
    return skip

def setup(app):
    app.connect('autodoc-skip-member', skip_tests)