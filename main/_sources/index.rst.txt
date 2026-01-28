splatkit
========

Welcome to **splatkit**!

splatkit is a modular toolkit for Gaussian Splatting training.
Wrapping `gsplat <https://github.com/nerfstudio-project/gsplat>`_ 's 
powerful rendering primitives, it provides an engineering-focused abstraction 
layer that (hopefully) makes your code easier to write and reason about. 

What is Gaussian Splatting?
----------------------------

Gaussian Splatting is a fast neural rendering technique that represents scenes as 
collections of 3D Gaussians. When rendered, these fuzzy blobs get projected onto 
your screen and alpha-composited into photo-realistic images. It's like pointillism, 
but with math that actually makes sense—and it's fast enough to train and iterate on interactively.

Why splatkit?
-------------

**Built on gsplat**
  Standing on the shoulders of giants (and their highly optimized CUDA kernels). 
  SplatKit handles the boilerplate so you can focus on the fun parts.

**Reproducibility & Clarity**
  Clear abstractions and configuration objects make your experiments easier to 
  understand, share, and reproduce. No more digging through nested notebooks trying 
  to remember which magic number you changed at 3 AM.

**Rapid Experimentation**
  Swap renderers, loss functions, or densification strategies in seconds without 
  rewriting your training loop. Write new modules with minimal boilerplate to test 
  your research ideas faster.

**Production Readiness**
  The same modular design that accelerates research also ensures consistent, 
  reproducible training for production deployments. Checkpoint management, distributed 
  training, and built-in evaluation metrics mean your experiments transition smoothly 
  from prototype to production.

Next Steps
----------

- Head to :doc:`installation` to get started
- Check out :doc:`quickstart` for a complete training example
- When you're ready to extend splatkit, see the :doc:`customization/index` guide

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Customization
   :hidden:

   customization/modules
   customization/renderers
   customization/loss_functions
   customization/data_providers

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/trainer
   api/renderer
   api/loss_fn
   api/data_provider
   api/splat
   api/modules/index

.. toctree::
   :maxdepth: 2
   :hidden:
