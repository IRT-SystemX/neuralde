.. _doc_techniques:

📚 Technical docs
======================================================


The classes of neuralde are simple to use. Create an instance and apply the method :func:`transform`.
The prediction methods from the documentations ensures the
compliance of models from various ML/DL libraries (such as Keras and scikit-learn) to **neuralde**.

.. autoclass:: neural_de.transformations.transformation.BaseTransformation
   :noindex:

.. autoclass:: neural_de.transformations.brightness_enhancer.BrightnessEnhancer
   :noindex:

.. autoclass:: neural_de.transformations.centered_zoom.CenteredZoom
   :noindex:

.. autoclass:: neural_de.transformations.de_rain_enhancer.DeRainEnhancer
   :noindex:

.. autoclass:: neural_de.transformations.de_snow_enhancer.DeSnowEnhancer
   :noindex:

.. autoclass:: neural_de.transformations.kernel_deblurring_enhancer.KernelDeblurringEnhancer
   :noindex:

.. autoclass:: neural_de.transformations.night_image_enhancer.NightImageEnhancer
   :noindex:

.. autoclass:: neural_de.transformations.resolution_enhancer.ResolutionEnhancer
   :noindex:

.. autoclass:: neural_de.transformations.transformation_pipeline.TransformationPipeline
   :noindex:

.. autoclass:: neural_de.transformations.diffusion.diffusion_enhancer.DiffusionEnhancer
   :noindex:


