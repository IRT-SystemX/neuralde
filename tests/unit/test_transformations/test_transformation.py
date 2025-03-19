import logging
import numpy as np
from neural_de.transformations.transformation import BaseTransformation


class TestTransformation:
    def test_init(self):
        # all transformations should have a logger
        tf = BaseTransformation()
        assert hasattr(tf, "_logger")
        assert isinstance(tf._logger, logging.Logger)

