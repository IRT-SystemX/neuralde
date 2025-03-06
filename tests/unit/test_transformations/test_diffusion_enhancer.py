from neural_de.transformations import DiffusionEnhancer, DiffPureConfig
import numpy as np
import torch
from pytest import raises

class TestDiffusionEnhancer:

    def test_transform_diffusion(self):
        # Images source
        image = np.ones(shape=(256, 256, 3), dtype=np.float32)
        image[:, :, 0] *= 0.147
        image[:, :, 1] *= 0.369
        image[:, :, 2] *= 0.258
        image2 = np.ones(shape=(256, 256, 3), dtype=np.float32)
        image2[:, :, 0] *= 0.147
        image2[:, :, 1] *= 0.369
        image2[:, :, 2] *= 0.258
        imageNotNorm = np.ones(shape=(256, 256, 3), dtype=np.uint8)
        imageNotNorm[:, :, 0] *= 70
        imageNotNorm[:, :, 1] *= 150
        imageNotNorm[:, :, 2] *= 220
        # Test
        # Initiate config
        config = DiffPureConfig()
        config.t = 10
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        diff_ehn = DiffusionEnhancer(device=device, config=config)
        transformed_image = diff_ehn.transform([image])
        transformed_imageNotNorm = diff_ehn.transform([imageNotNorm])
        # Test type of result
        assert isinstance(transformed_image[0], np.ndarray)
        # Test dimensions of the images
        assert (transformed_image[0].shape == (256, 256, 3))
        # Batch test
        transformed_image2 = diff_ehn.transform([image, image2])
        assert (len(transformed_image2) == 2)
        # Tests inputs are normalized or not
        assert (np.max(np.abs(transformed_image[0] - image)) < 0.1)
        assert (np.max(np.abs(transformed_imageNotNorm[0] - (imageNotNorm/255))) < 0.1)
        # Empty input check
        img_batch = []
        with raises(ValueError):
            _ = diff_ehn.transform(img_batch)
        # Content not an image or with incorrect number of dimensions should raise an error
        with raises(ValueError):
            _ = diff_ehn.transform([["not_an_image"]])
        with raises(ValueError):
            _ = diff_ehn.transform(np.zeros((3, 3, 3, 3, 3)))
        with raises(ValueError):
            _ = diff_ehn.transform(np.zeros((3, 3)))