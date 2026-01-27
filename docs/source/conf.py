# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'Splatkit'
copyright = '2026, Feiyou Guo'
author = 'Feiyou Guo'
username = 'feiyuguo'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = ''

try:
    from splatkit import __version__
    version = __version__
    release = __version__
except ImportError:
    version = 'dev'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.duration',
    'sphinx_copybutton',
    'myst_parser',
]

# MyST parser configuration
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'fieldlist',
]

templates_path = ['_templates']
exclude_patterns = []

# Support both .rst and .md files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
html_title = f"{project} Documentation"

# Furo theme options
html_theme_options = {
    "top_of_page_button": "edit",
    "source_repository": f"https://github.com/{username}/splatkit",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "announcement": None,
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#2563eb",
        "font-stack": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif",
        "font-stack--monospace": "'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace",
        # Increase font sizes
        "font-size--normal": "17px",
        "font-size--small": "15px",
        "font-size--small--2": "14px",
        "font-size--small--3": "13px",
        "font-size--small--4": "12px",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#60a5fa",
        # Same larger font sizes for dark mode
        "font-size--normal": "17px",
        "font-size--small": "15px",
        "font-size--small--2": "14px",
        "font-size--small--3": "13px",
        "font-size--small--4": "12px",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": f"https://github.com/{username}/splatkit",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Extension configuration -------------------------------------------------
# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Napoleon settings
napoleon_google_style = True
napoleon_numpy_style = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Mock imports for packages that require CUDA or are not available during docs build
autodoc_mock_imports = ['fused_ssim']