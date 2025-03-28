from pathlib import Path

import cv2
import numpy as np
import torch

from neural_de.transformations.diffusion.diffusion_enhancer import DiffusionEnhancer, DiffPureConfig

TESTS_DIR = Path(__file__).parent.parent.parent.resolve()


def test_transform():
    # assert TARGET_MODEL.is_file(), "Model not available"
    origin_path = Path(TESTS_DIR / "regression/data/diffpure/in.png")
    origin_image = cv2.imread(str(origin_path))
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    result_path = Path(TESTS_DIR / 'regression/data/diffpure/out_car.npz')
    result_image = np.load(str(result_path))["arr_0"]
    result_image = np.resize(result_image, (256,256,3))
    config = DiffPureConfig()
    config.t = 1
    enhancer = DiffusionEnhancer(config=config)
    #tensor = torch.Tensor([origin_image])
    purified = enhancer.transform([origin_image])[0] #* 255

    assert np.mean(abs((purified).astype(np.uint8) - result_image.astype(np.uint8))) < 0.01
