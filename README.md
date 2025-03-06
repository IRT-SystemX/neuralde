<div align="center">
    <img src="docs/assets/Logo_ConfianceAI.png" width="20%" alt="ConfianceAI Logo" />
    <h1 style="font-size: large; font-weight: bold;">neural_de</h1>
</div>
<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.10-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
    <!-- Pylint Score Badge -->
    <a href="http://confianceai.pages.irt-systemx.fr/ec_4/neuralde/pylint/index.html">
        <img src="http://confianceai.pages.irt-systemx.fr/ec_4/neuralde/pylint/pylint.svg" alt="Pylint Score">
    </a>
    <!-- Flake8 Report Badge -->
    <a href="http://confianceai.pages.irt-systemx.fr/ec_4/neuralde/flake8/index.html">
        <img src="http://confianceai.pages.irt-systemx.fr/ec_4/neuralde/flake8/flake8.svg" alt="Flake8 Report">
    </a>
    <!-- Coverage Badge -->
    <a href="http://confianceai.pages.irt-systemx.fr/ec_4/neuralde/coverage/index.html">
        <img src="http://confianceai.pages.irt-systemx.fr/ec_4/neuralde/coverage/coverage.svg" alt="Code Coverage">
    </a>
</div>





<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.9-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
</div>
<br>

<div align="center">
  <a href="">Quickstart</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="">Docs</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="">Examples</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://www.confiance.ai/">Confiance.ai</a>
&nbsp;
  <hr />
</div>

# NEURAL_DE
## Description
NeuralDe is a library made to improve the robustness of your models at test time. It proposes a set of methods 
that will allow you to remove identified corruptions in your data before you send it to your model. 
This library addresses issues such as meteorological corruptions, distribution shifts etc.

All methods provided by the library are associated with examples based on open sourced images. 
You can refer to each jupyter notebook for specifics.
These notebooks are linked directly in the technical documentation.

## User Guidelines

## Setup your environment
Validated python version : 3.9.18

Linux setting:

```
pip install virtualenv
virtualenv myenv
source myenv/bin/activate
```

Windows setting:

```
pip install virtual env 
virtualenv myenv 
.\myenv\Scripts\activate
```

## Installation
If you are installing it from source please instead refers to the section 
"Installation from source".
````
pip install neural_de
pip show neural_de
````
## Identity card
See the [component identity card](identity_card.yml) for a synthetic view of its properties.
## Required Hardware
All methods have been tested on CPU and GPU. 

## Available methods in release 1.0.0
* [Resolution_Enhancer](neural_de.transformations.rst#resolution-enhancer-label) : enhance image resolution (GPU compat)
* [NightImageEnhancer](neural_de.transformations.rst#night-image-enhancer-label) : transform night images into daylight ones (GPU compat)
* [KernelDeblurringEnhancer](neural_de.transformations.rst#kernel-deblurring-enhancer-label) : Improve blurry images
* [DeSnowEnhancer](neural_de.transformations.rst#desnow-enhancer-label) : Removes snow from images (GPU compat)
* [DeRainEnhancer](neural_de.transformations.rst#derain-enhancer-label) : Removes rain from images (GPU compat)
* [BrightnessEnhancer](neural_de.transformations.rst#brightness-enhancer-label) : Improve luminosity
* [CenteredZoom](neural_de.transformations.rst#centered-zoom-label) : Zoom in the middle of the image, with a given zoom ratio.
* [DiffusionEnhancer](neural_de_transformations.rst#diffusion-enhancer-label) : Purify noise on images and increase robustness against attack.

## Usage

All methods in neurelDE follow the same syntax.
```
from neural_de.transformations import <Method>
method = <Method()>
transformed_image_batch = method.transform(image_batch)
```

### Detailed examples with notebooks

The neuralDE library provides several methods to preprocess images.
To understand how to use these methods, click on the link to see the following notebooks.
* [Brightness_Enhancer](./examples/Brightness_Enhancered_examples.ipynb)
: Notebook to present how to use the Brightness_Enhancered class. This method allows us to brighten images.
* [CenteredZoom](./examples/CenteredZoom_example.ipynb)
: Notebook to present how to use the Centered_zoom class. This method allows us to zoom on the center of images.
* [Derain_Enhancer](./examples/DeRainEnhancer_example.ipynb)
: Notebook to present how to use the Derain_Enhancer class. This method allows us to remove the rain on images.
* [Desnow_Enhancer](./examples/SnowRemoval.ipynb)
: Notebook to present how to use the Desnow_Enhancer class. This method allows us to remove the snow on images.
* [Kernel_Deblurring_Enhancer](./examples/KernelDeblurringEnhancer.ipynb)
: Notebook to present how to use the Kernel_Deblurring class. This method allows us to deblur the images.
* [Night_Image_Enhancer](./examples/NightEnhancer_example.ipynb)
: Notebook to present how to use the Night_Image_Enhancer class. This method allows us to improve the clarity of night images.
* [Resolution_Enhancer](./examples/ResolutionEnhancer_example.ipynb)
: Notebook to present how to use the Resolution_Enhancer class. This method allows us to increase the resolution of an zoomed image.
* [Diffusion_Enhancer](./examples/DiffpurEnhancer_example.ipynb)
: Notebook to present how to use the Diffusion_Enhancer class. This method allows us to purify noise into images. 

### Good practises and scientific guidelines

To guide you through the specifics and the best practises to implement this approach you'll find dedicated
guidelines [here](pdf/NeuralDE_Confiance.ai_Methodological_Guideline_v2.0.pdf)

## Description of Inputs and Outputs
NeuralDE is an **Image2Image library**, thus it is made to process **images batches**. 
It is composed of a list of classes called "enhancers". Each enhancer will have specific options that 
will be defined by the user at object initialisation. 

The process is simple: import the required classes from the library in your code and pass them their specific 
configurations using the parameters on initialisation. 

All the methods using computationally-expensive models (based on different kinds of neural networks
architectures, mostly CNN or transformer-based architectures) can be run on GPU.
To do so, set the parameter device="cuda" at class instantiation.
The concerned which can be run on GPU are:
- NightImageEnhancer
- DesnowEnhancer
- DerainEnhancer
- ResolutionEnhancer
- DiffusionEnhancer

Please refer to the html documentation of each method for more detailed informations.
### Installation from source
To install neuralde directly from the source repository :
- In a terminal in the project root directory (the one containing ./docs and ./neural_de), type :
  - Ensure you do have wheel installed in your virtual env. If not, install it (pip install wheel).
```
 python setup.py bdist_wheel
```
It will create in ./dist a wheel with all the dependency for neuralde.
- Then in you desired python environment, install the wheel with pip :
```
 pip install PATH_TO_THE_WHEEL/neuralde-0.1-py3-none-any.whl
```

## How to recompile the doc sphinx

If you need to reload the generation of the sphinx documentation, you need to install some packages.
To begin, install the sphinx packages:
'''
pip install --upgrade sphinx myst-parser[sphinx]
pip install nbsphinx
pip install autotyping
pip install pandoc
winget install --source winget --exact --id JohnMacFarlane.Pandoc
'''

In your file "conf.py", in the list of extensions, add "myst-parser" and "nbsphinx".
To generate the documentation, go to the "docs" directory and run this command in your terminal:
'''
.\make html
'''
To finish, reload your IDE.
