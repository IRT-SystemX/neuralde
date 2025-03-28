"""
Transformation pipeline for automation of multiple transformations methods
"""
import logging
from pathlib import Path
from typing import Union

import numpy as np
import yaml

from neural_de.transformations.transformation import BaseTransformation
# from neural_de import transformations
from neural_de.utils.twe_logger import log_and_raise
import importlib


def camel_to_snake(s):
    """ This function convert input strings s from Camelcase format to snake case format"""
    return ''.join(['_' + c.lower() if c.isupper() else c for c in s]).lstrip('_')


class TransformationPipeline(BaseTransformation):
    """
    Provides a pipeline object, to facilitate the automation of multiple transformations methods,
    and/or offer loading from a yaml file.

    You can check the example notebook **examples/Pipeline_example.ipynb** for details on the syntax
    and usage.
    An example of valid config file can be found in **examples/config/conf_user.yaml**

    Args:
        config: either a path toward a yaml configuration file, or a list of dict.
        logger: It is recommended to use the confiance.ai logger, obtainable with
            neural_de.utils.get_logger(...). If None, one logging with stdout will be provided.
    """
    def __init__(self, config: Union[str, list, Path], logger: logging.Logger = None):
        super().__init__(logger)
        if isinstance(config, list):
            self._pipeline_conf = config
        else:
            # Avoid lazy init for the configuration in order to check consistency asap.
            self._pipeline_conf = self._read_config(config)
        self._pipeline = None
        self._logger.info("Config file loaded")

    def _read_config(self, config_path: str) -> dict:
        """
        Read user configuration of pipeline that contains different transformations parameters

        Args:
            config_path: path to a yaml configuration file. See example/Pipeline_example for syntax
                specification
        Returns:
            The loaded configuration
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as yaml_file:
                config = yaml.safe_load(yaml_file)
        except FileNotFoundError:
            log_and_raise(self._logger, FileNotFoundError,
                          f"Config file '{config_path}' not found.")
        except yaml.YAMLError as e:
            log_and_raise(self._logger, yaml.YAMLError, f"Error parsing YAML file: {e}")
        self._logger.info("Config loaded from %s loaded", config_path)
        return config

    def _init_pipeline(self) -> None:
        """
        Initialize every transformation method in the pipeline (once on first transform call).
        """
        self._logger.info("Loading all the pipeline methods and models")
        self._pipeline = []
        try:
            # for every transformation, initialize it and store the resulting instance
            # initialization can be costly, so it is done only once, during first transform call.
            for transformation_conf in self._pipeline_conf:
                transformation_name: str = transformation_conf['name']
                parameters = transformation_conf.get('init_param', {})

                # Get module name corresponding to tranformation

                module_transformation = importlib.import_module("neural_de.transformations." +
                                                                camel_to_snake(transformation_name))

                transformation = getattr(module_transformation, transformation_name)
                self._pipeline.append(transformation(**parameters))
        except KeyError:
            log_and_raise(self._logger, KeyError, "Invalid structure for method " +  transformation_conf["name"])
        except AttributeError:
            log_and_raise(self._logger, AttributeError, "Transformation " + transformation_name +
                          "not found in neural.transformations")
        except TypeError:
            log_and_raise(self._logger, TypeError, "Invalid call during initialization of " +
                          transformation_conf["name"])
        self._logger.info("All pipeline models successfully loaded")

    def transform(self, images: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
        """
        Sequentially apply every method of the pipeline on a batch of image, and returns
        the resulting images.

        Args:
          images: Batch of images. Each image should be of a ``np.ndarray`` of target_shape *(h,w,
            channels)*
        Returns:
          Resulting batch of images, one per image provided.
        """
        # verify if image is a valid batch
        self._check_batch_validity(images)
        # Lazy method init, as it can be costly both in term of computing power and ram.
        if self._pipeline is None:
            self._init_pipeline()
        try:
            # For each method in the pipeline
            for i, transformation in enumerate(self._pipeline):
                self._logger.info("Applying method %s to images", transformation)
                # Retrieves optional transform() parameters if any
                transformation_parameters = self._pipeline_conf[i]\
                    .get('transform', {})

                # apply the transformation to the images
                images = transformation.transform(images=images, **transformation_parameters)
        except TypeError as e:
            raise log_and_raise(self._logger, TypeError,
                                f"Invalid call during function transform of "
                                f"'{transformation}': {e}")
        return images
