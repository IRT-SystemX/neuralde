import numpy as np
from pathlib import Path
from neural_de.transformations import TransformationPipeline
from unittest.mock import MagicMock
from pytest import raises
from neural_de.transformations import ResolutionEnhancer


TESTS_FOLDER = Path(__file__).parent.parent.parent.resolve()
CONFIG_PATH = Path(TESTS_FOLDER / 'unit/config/conf_test.yaml')


class TestTransformationPipeline:
    def test_init(self):
        # test with invalid configuration input
        logger = MagicMock()
        logger.info = MagicMock()
        with raises(TypeError):
            _ = TransformationPipeline(config=dict(), logger=logger)

        # realistic call test with list input
        ref_conf = [{"name": 'ResolutionEnhancer', "init_param": {'device': 'cpu'},
                     "transform": {'target_shape': [5, 5], 'crop_ratio': 0.}}]
        logger = MagicMock()
        logger.info = MagicMock()
        pipeline = TransformationPipeline(config=ref_conf, logger=logger)
        assert ref_conf == pipeline._pipeline_conf
        assert logger.info.call_count > 0

    def test_read_config(self):
        # input validation test
        ref_conf = "not_an_existing_file.yaml"
        with raises(FileNotFoundError):
            _ = TransformationPipeline(config=ref_conf)

        # realistic call test
        logger = MagicMock()
        logger.info = MagicMock()
        ref_conf = [{"name": 'ResolutionEnhancer', "init_param": {'device': 'cpu'},
                     "transform": {'target_shape': [5, 5], 'crop_ratio': 0.}}]
        pipeline = TransformationPipeline(config=CONFIG_PATH, logger=logger)
        assert ref_conf == pipeline._pipeline_conf
        assert logger.info.call_count > 0

    def test_init_pipeline(self):
        # verify it raises an keyerror when the config structure is invalid
        pipeline = TransformationPipeline(config=CONFIG_PATH)
        pipeline._pipeline_conf = [{}]
        with raises(KeyError):
            pipeline._init_pipeline()

        # verify it raises an attributeerror when the method is not a neuralde transformation
        pipeline._pipeline_conf = [{"name":"Not_a_neural_de_method", "init_params": {}}]
        with raises(AttributeError):
            pipeline._init_pipeline()

        # verify it raises a TypeError when called with invalid parameter values
        ref_conf = [{"name": 'ResolutionEnhancer', "init_param": {'not_a_valid_param': 'cpu'},
                         "transform": {'target_shape': [5, 5], 'crop_ratio': 0.}}]
        pipeline = TransformationPipeline(config=ref_conf)
        with raises(TypeError):
            pipeline._init_pipeline()

        # verify init_pipeline is able to instantiate a valid transformation
        pipeline = TransformationPipeline(config=CONFIG_PATH)
        pipeline._init_pipeline()
        assert len(pipeline._pipeline) == 1
        assert isinstance(pipeline._pipeline[0], ResolutionEnhancer)

    def test_transformation(self):
        # calling with an invalid  batch should raise a typeerror
        pipeline = TransformationPipeline(config=[], logger=MagicMock())
        pipeline._pipeline = []
        with raises(ValueError):
            pipeline.transform(np.array((3,3,11,11,5)))

        # test pipeline with a single method
        img = np.ndarray((3, 10, 10, 3))
        transformation = MagicMock()
        transformation.transform = MagicMock(return_value=np.ndarray((3, 11, 11, 1)))
        pipeline = TransformationPipeline(config=CONFIG_PATH)
        pipeline._pipeline = [transformation]
        res_img = pipeline.transform(images=img)
        assert res_img.shape == (3, 11, 11, 1)  # check target_shape parameter
        assert transformation.transform.call_count == 1

        # test pipeline with multiple methods
        img = np.ndarray((3, 10, 10, 3))
        transformation = MagicMock()
        transformation.transform = MagicMock(return_value=np.ndarray((3, 11, 11, 1)))
        ref_conf = [{"name": 'ResolutionEnhancer', "init_param": {'device': 'cpu'},
                     "transform": {'target_shape': [5, 5], 'crop_ratio': 0}},
                    {"name": 'ResolutionEnhancer', "init_param": {'device': 'cpu'},
                     "transform": {'target_shape': [5, 5], 'crop_ratio': 0}}]
        pipeline = TransformationPipeline(config=ref_conf)
        pipeline._pipeline = [transformation, transformation]
        res_img = pipeline.transform(images=img)
        assert res_img.shape == (3, 11, 11, 1)  # check target_shape parameter
        assert transformation.transform.call_count == 2 # should be called 2 times
