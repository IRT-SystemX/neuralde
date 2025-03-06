import numpy as np
from pathlib import Path
from neural_de.transformations import (TransformationPipeline, DeSnowEnhancer, KernelDeblurringEnhancer,
                                       NightImageEnhancer, ResolutionEnhancer, DeRainEnhancer)
import itertools

TESTS_FOLDER = Path(__file__).parent.parent.parent.resolve()


class TestTransformationPipeline:
    def test_all_combinations(self):
        # test that all combination of methods can be called
        transformations = {
            NightImageEnhancer: {},
            DeRainEnhancer: {},
            DeSnowEnhancer: {},
            KernelDeblurringEnhancer: {},
            ResolutionEnhancer: {"target_shape": [34,33]}
        }
        for pair in itertools.combinations(transformations.keys(),2):
            conf = [{"name": pair[0].__name__, "transform":transformations[pair[0]]},
                    {"name": pair[1].__name__, "transform": transformations[pair[1]]}]
            images = np.ndarray((1, 34, 33, 3))
            pipeline = TransformationPipeline(conf)
            out_images = pipeline.transform(images)
            assert out_images[0].shape == (34, 33, 3)

    def test_pipeline_yaml_simple(self):
        config = Path(TESTS_FOLDER / 'integration/config/test_conf_pipeline_1.yaml')
        batch_img = np.ndarray((3, 10, 10, 3))

        # compute reference results
        model = DeSnowEnhancer()
        image_transformation_1 = model.transform(batch_img)
        model = KernelDeblurringEnhancer(kernel="medium")
        ref_image = model.transform(image_transformation_1)

        # pipeline call
        caller = TransformationPipeline(config)
        res_img = caller.transform(images=batch_img)

        assert np.array_equal(res_img, ref_image)

    def test_pipeline_yaml_all_methods(self):
        config = Path(TESTS_FOLDER / 'integration/config/test_conf_pipeline_2.yaml')
        batch_img = np.ndarray((3, 64, 64, 3))

        # ref results
        model = NightImageEnhancer()
        image_transformation = model.transform(images=batch_img)
        model = DeRainEnhancer(device="cpu")
        image_transformation = model.transform(images=image_transformation)
        model = DeSnowEnhancer(device="cpu")
        image_transformation = model.transform(images=image_transformation)
        model = KernelDeblurringEnhancer(kernel="medium")
        image_transformation = model.transform(images=image_transformation)
        model = ResolutionEnhancer(device="cpu")
        ref_image = model.transform(images=image_transformation, target_shape=[64, 32],
                                    crop_ratio=.25)

        # pipeline call
        caller = TransformationPipeline(config)
        res_img = caller.transform(images=batch_img)

        assert np.array_equal(res_img, ref_image)
