Installation
============

Welcome to splatkit!

**splatkit** is a modular toolkit for 3D and 2D Gaussian Splatting, providing flexible components for training, rendering, and visualization.

Features
--------

* Default modules provided for simple 3DGS and 2DGS training
* Modular architecture with pluggable components for customer renderer, loss function, data provider, and more
* Built-in COLMAP data provider
* Real-time visualization with nerfview
* Distributed training support

Requirements
------------

* Python 3.10 or higher
* **PyTorch 2.0+ with CUDA support** (must be installed separately)
* CUDA-capable GPU (for training)

.. important::
   **splatkit does not include PyTorch as a dependency.** You must install PyTorch with 
   CUDA support separately before installing splatkit.

Basic Installation
------------------

**Step 1: Install PyTorch with CUDA**

First, install PyTorch with CUDA support. Visit `pytorch.org <https://pytorch.org/get-started/locally/>`_ 
or use one of these examples:

.. code-block:: bash

   # PyTorch 2.2.2 with CUDA 12.1 (recommended)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   
   # Or PyTorch with CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

**Step 2: Install fused-ssim (Optional but Recommended)**

For improved training performance with optimized SSIM computation:

.. code-block:: bash

   pip install git+https://github.com/rahul-goel/fused-ssim@98126b7781f9e563234c92d2bf08ee0994f4f175

.. note::
   **fused-ssim** provides a CUDA-accelerated SSIM loss implementation that significantly 
   improves training speed and quality. While optional, it is **highly recommended** for 
   optimal performance. If not installed, a fallback PyTorch-based SSIM implementation 
   will be used automatically, though it may be slower.

**Step 3: Install splatkit**

Clone the repository and install the core package:

**Using pip:**

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/splatkit.git
   cd splatkit
   pip install -e .

**Using uv:**

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/splatkit.git
   cd splatkit
   uv pip install -e .

This installs splatkit with core dependencies only (PyTorch not included).

Using splatkit as a Library
----------------------------

If you want to use splatkit as a dependency in your own project, you can add it with **uv**:

.. code-block:: bash

   # First, ensure PyTorch with CUDA is installed
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   
   # Install fused-ssim for better training performance (recommended)
   pip install git+https://github.com/rahul-goel/fused-ssim@98126b7781f9e563234c92d2bf08ee0994f4f175
   
   # Add splatkit to your project
   uv add git+https://github.com/YOUR_USERNAME/splatkit.git
   
   # Or add with specific features
   uv add "splatkit[all] @ git+https://github.com/YOUR_USERNAME/splatkit.git"

This will add splatkit to your project's ``pyproject.toml`` and install it in your environment.

Optional Dependencies
---------------------

Install with specific features based on your needs. Use either **pip** or **uv**:

Fused SSIM (Highly Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For significantly faster training with optimized SSIM computation:

.. code-block:: bash

   pip install git+https://github.com/rahul-goel/fused-ssim@98126b7781f9e563234c92d2bf08ee0994f4f175

.. tip::
   This is technically optional, but **highly recommended** for production use. The fused SSIM 
   implementation provides:
   
   - **Faster training**: CUDA-optimized SSIM computation
   - **Better quality**: More accurate structural similarity measurements
   - **Lower memory usage**: Efficient GPU memory utilization
   
   Without it, a fallback PyTorch-based SSIM implementation will be used automatically.

Real-time Viewer
~~~~~~~~~~~~~~~~

For interactive visualization with nerfview:

.. code-block:: bash

   # Using pip
   pip install -e ".[viewer]"
   
   # Using uv
   uv pip install -e ".[viewer]"

Progress Tracking
~~~~~~~~~~~~~~~~~

For progress bars during training:

.. code-block:: bash

   # Using pip
   pip install -e ".[progress]"
   
   # Using uv
   uv pip install -e ".[progress]"

TensorBoard Logging
~~~~~~~~~~~~~~~~~~~

For TensorBoard integration:

.. code-block:: bash

   # Using pip
   pip install -e ".[tensorboard]"
   
   # Using uv
   uv pip install -e ".[tensorboard]"

All Features
~~~~~~~~~~~~

Install everything:

.. code-block:: bash

   # Using pip
   pip install -e ".[all]"
   
   # Using uv
   uv pip install -e ".[all]"

Development Installation
------------------------

For contributors, install with development tools:

.. code-block:: bash

   # Using pip
   pip install -e ".[all-dev]"
   
   # Using uv
   uv pip install -e ".[all-dev]"

This includes testing, linting, formatting tools, and documentation dependencies.

Available Options
-----------------

* ``viewer`` - Real-time visualization (nerfview, viser, splines)
* ``progress`` - Progress bars (tqdm)
* ``tensorboard`` - TensorBoard logging
* ``dev`` - Development tools (pytest, black, mypy)
* ``docs`` - Documentation building (Sphinx, Furo)
* ``all`` - All optional features
* ``all-dev`` - All features + development tools

Verify Installation
-------------------

To verify the installation:

.. code-block:: python

   import splatkit
   print(splatkit.__version__)

You should see the version number printed without any errors.

To check if fused-ssim is installed:

.. code-block:: python

   try:
       from fused_ssim import fused_ssim
       print("✓ fused-ssim is installed (recommended)")
   except ImportError:
       print("✗ fused-ssim not found. Install for better performance:")
       print("  pip install git+https://github.com/rahul-goel/fused-ssim@98126b7...")

Troubleshooting
---------------

**Missing fused-ssim warning during training**

If you see a warning like:

.. code-block:: text

   WARNING: fused-ssim is not installed, using ssim instead

This is normal - a fallback SSIM implementation will be used. For better performance, install fused-ssim:

.. code-block:: bash

   pip install git+https://github.com/rahul-goel/fused-ssim@98126b7781f9e563234c92d2bf08ee0994f4f175

**CUDA out of memory errors**

If you encounter CUDA out of memory errors:

- Reduce batch size in your data provider configuration
- Use smaller image resolutions for training
- Ensure no other processes are using GPU memory

**PyTorch/CUDA version mismatch**

Ensure your PyTorch installation matches your CUDA version:

.. code-block:: python

   import torch
   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
