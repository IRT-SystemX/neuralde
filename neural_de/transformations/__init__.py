"""
Module with the main images transformations methods of the neural_de library.
You can find more on how to use any of the proposed method in ``./examples``, or in the method's
class documentation.

List of the available methods :
    - ResolutionEnhancer: enhance image resolution
    - NightImageEnhancer: transform night images into daylight ones
    - KernelDeblurringEnhancer: Improve blurry images
    - DeSnowEnhancer: Removes snow from images
    - DeRainEnhancer: Removes rain from images
    - BrightnessEnhancer: Improves image brightness
    - CenteredZoom: Centered crop of an image at a given ratio
    - DiffusionEnhancer : Enhance the image using diffusion-based denoising

Special methods :
    - TransformationPipeline : Allows the automation of any combination of the previous methods,
      and loading from file.
"""
