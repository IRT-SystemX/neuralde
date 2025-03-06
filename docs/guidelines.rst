.. _README:

📖 Guidelines:
=================================

-  `📝 Description <#description>`__
-  `🔍 Specifications <#specifications>`__
-  `🚀 Quick Start <#Quick Start>`__
-  `🎮 Usage <#usage>`__
-  `🔀 Description of Inputs and Outputs <#format>`__
-  `💻 Required Hardware <#hardware>`__
-  `📚 References <#references>`__


.. raw:: html

   <div id='description'/>

📝 Description
--------------

This library offers *neuralde methods* to help proving ownership
on the machine learning models you train.

NeuralDe is a library made to improve the robustness of your models at test time. It proposes a set of methods
that will allow you to remove identified corruptions in your data before you send it to your model.
This library addresses issues such as meteorological corruptions, distribution shifts etc.

All methods provided by the library are associated with examples based on open sourced images.
You can refer to each jupyter notebook for specifics.
These notebooks are linked directly in the technical documentation.

.. raw:: html

   <div id='specifications'/>

🔍 Specifications
-----------------

-  Version: 1.0.0
-  Python Version: python 3.9
-  Strong Dependencies: tensorflow, pytorch
-  Thematic:
-  Trustworthy:
-  Hardware : CPU / GPU
-  Engineering activities : 
-  Scientific challenge: 
-  Functional set :  


.. raw:: html

   <div id='Quick Start'/>

🚀 Quickstart
--------------

To install and use the neural_de library, it is recommended to create a
Python virtual environment. You can do that with virtualenv, as follows:

Setting environement
~~~~~~~~~~~~~~~~~~~~

Linux setting:

.. code:: bash

   pip install virtualenv
   virtualenv myenv
   source myenv/bin/activate

Windows setting:

.. code:: bash

   pip install virtual env
   virtualenv myenv
   .\myenv\Scripts\activate

Installation
~~~~~~~~~~~~

Once your virtual environment is activated, you can install the neural_de
library. In the confiance environnement:

.. code:: bash

   git clone https://git.irt-systemx.fr/confianceai/ec_4/neuralde.git
   cd neuralde
   pip install .

or install the component from Nexus

.. code:: bash

   pip install neuralde
   pip show neuralde

This command will install the uqmodels package and all required
dependencies.

.. raw:: html

   <div id='usage'/>

🎮 Usage
----------------------------

NeuralDe is a library made to improve the robustness of your models at test time. It proposes a set of methods
that will allow you to remove identified corruptions in your data before you send it to your model.
This library addresses issues such as meteorological corruptions, distribution shifts etc.

All methods provided by the library are associated with examples based on open sourced images.
You can refer to each jupyter notebook for specifics.
These notebooks are linked directly in the technical documentation.

All methods in neuralde follow the same syntax

.. code:: python

   from neural_de.transformations import <Method>
   method = <Method()>
   transformed_image_batch = method.transform(image_batch)



For more detailed usage and explanations of the different options and
settings available in neural_de, please refer to the full
`documentation <mettre%20le%20lien%20de%20la%20documentation>`__.

📚 For more practical examples and detailed usage scenarios, please refer
to the Examples directory in our `GitLab
repository <https://git.irt-systemx.fr/confianceai/ec_4/neuralde.git>`__.
These examples provide a variety of use cases that can help you
understand how to utilize ml_watermarking in your own projects.

.. raw:: html

   <div id='io'/>

🔀 Format
------------------------------------

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

.. raw:: html

   <div id='hardware'/>

💻 Hardware
----------------------------------------

The neuralde library is designed to run efficiently on modern CPU
architectures and GPU. The computational load involved in the tasks performed by
this library needs an GPU.


.. raw:: html

   <div id='references'/>

📚 References
---------------------------

To display the component's references, `Open the PDF file <_static/test_docs_pdf.pdf>`_
