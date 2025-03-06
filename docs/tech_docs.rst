.. _doc_techniques:

📚 Technical docs
======================================================


The classes of neuralde are simple to use. Create an instance and apply the method :func:`transform`.
The prediction methods from the documentations ensures the
compliance of models from various ML/DL libraries (such as Keras and scikit-learn) to **neuralde**.

.. autoclass:: neural_de.transformations._transformation.BaseTransformation
   :noindex:

.. autoclass:: neural_de.transformations._brightness_enhancer.BrightnessEnhancer
   :noindex:

.. autoclass:: neural_de.transformations._centered_zoom.CenteredZoom
   :noindex:

.. autoclass:: neural_de.transformations._derain_enhancer.DeRainEnhancer
   :noindex:

.. autoclass:: neural_de.transformations._desnow_enhancer.DeSnowEnhancer
   :noindex:

.. autoclass:: neural_de.transformations._kernel_deblurring_enhancer.KernelDeblurringEnhancer
   :noindex:

.. autoclass:: neural_de.transformations._night_image_enhancer.NightImageEnhancer
   :noindex:

.. autoclass:: neural_de.transformations._resolution_enhancer.ResolutionEnhancer
   :noindex:

.. autoclass:: neural_de.transformations._transformation_pipeline.TransformationPipeline
   :noindex:

.. autoclass:: neural_de.transformations._diffusion._diffusion_enhancer.DiffusionEnhancer
   :noindex:

The neuralDE library provides several methods to preprocess images.
To understand how to use these methods, click on the link to see the following notebooks.
Please refer to the notebooks in ./examples to for working examples:

.. toctree::
   :maxdepth: 2

Brightness_Enhancer (This method allows us to brighten images.)
`Notebook Brightness Enhancered <./examples/Brightness_Enhancered_examples.ipynb>`_

CenteredZoom (This method allows us to zoom on the center of images.)
`Notebook Centered Zoom <./examples/CenteredZoom_example.ipynb>`_

Derain_Enhancer (This method allows us to remove the rain on images.)
`Notebook DeRain Enhancer <./examples/DeRainEnhancer_example.ipynb>`_

Desnow_Enhancer (This method allows us to remove the snow on images.)
`Notebook DeSnow Enhancer <./examples/SnowRemoval.ipynb>`_

Kernel_Deblurring_Enhancer (This method allows us to deblur the images.)
`Notebook Kernel Deblurring <./examples/KernelDeblurringEnhancer.ipynb>`_

Night_Image_Enhancer (This method allows us to improve the clarity of night images.)
`Notebook Night Enhancer <./examples/NightEnhancer_example.ipynb>`_

Resolution_Enhancer (This method allows us to increase the resolution of an zoomed image.)
`Notebook Resolution Enhancer <./examples/ResolutionEnhancer_example.ipynb>`_

Diffusion_Enhancer (This method allows us to purify noise into images.)
`Notebook Diffusion Enhancer <./examples/DiffpurEnhancer_example.ipynb>`_
