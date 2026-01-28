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

**Step 2: Install splatkit**

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
   
   # Add splatkit to your project
   uv add git+https://github.com/YOUR_USERNAME/splatkit.git
   
   # Or add with specific features
   uv add "splatkit[all] @ git+https://github.com/YOUR_USERNAME/splatkit.git"

This will add splatkit to your project's ``pyproject.toml`` and install it in your environment.

Optional Dependencies
---------------------

Install with specific features based on your needs. Use either **pip** or **uv**:

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
