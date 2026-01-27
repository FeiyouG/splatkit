# Documentation Developer Guide

This guide is for developers working on splatkit's documentation.

## Documentation Structure

The main documentation page (`index.rst`) includes the project README.md, so any updates to the README will automatically appear in the documentation. This ensures consistency between the repository and docs.

## Building Locally

Install dependencies and build (using either pip or uv):

```bash
# Install documentation dependencies
pip install -r requirements.txt
# or
uv pip install -r requirements.txt

# Install splatkit (from project root)
pip install -e ..
# or
uv pip install -e ..

# Build the documentation
sphinx-build source _build/html
```

The built documentation will be in `_build/html/index.html`.

View it via:
- `open _build/html/index.html` (macOS)
- Or serve it: `cd _build/html && python -m http.server 8000` then open http://localhost:8000

## Theme and Features

The documentation now uses:

- **Furo theme** - Modern, clean design with automatic dark mode
- **sphinx-copybutton** - One-click code copying in code blocks
- **Improved content structure** - Better organized and more polished

## How GitHub Actions Builds Docs

The workflow (`.github/workflows/docs.yml`) uses `sphinx-build` directly:

```bash
cd docs
sphinx-build -b html source ../build/html
```