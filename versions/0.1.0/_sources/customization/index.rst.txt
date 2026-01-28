Custom Modules
===================

Learn how to extend splatkit with your own custom components.

.. toctree::
   :maxdepth: 2
   :caption: Customization Topics

   modules
   renderers
   loss_functions
   data_providers

Overview
--------

splatkit is designed to be modular and extensible. You can create custom:

- **Modules**: Add hooks into the training loop for logging, visualization, checkpointing, etc.
- **Renderers**: Implement new Gaussian splatting techniques or rendering algorithms
- **Loss Functions**: Design custom training objectives with novel regularization terms
- **Data Providers**: Load data from new formats or implement custom data augmentation

All custom components inherit from base classes and implement specific abstract methods.
The architecture ensures your custom components work seamlessly with the rest of the framework.

Quick Links
-----------

- :doc:`modules` - Most common customization, great for adding functionality to training
- :doc:`renderers` - For implementing new splatting algorithms  
- :doc:`loss_functions` - For custom training objectives
- :doc:`data_providers` - For loading data from new sources

Getting Help
------------

If you're unsure which component to customize:

- Want to add logging, visualization, or evaluation? → Create a **Module**
- Want to change how Gaussians are rasterized? → Create a **Renderer**
- Want to modify the training objective? → Create a **Loss Function**
- Want to load data from a new format? → Create a **Data Provider**
