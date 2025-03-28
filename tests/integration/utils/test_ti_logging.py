import os

from neural_de.transformations.kernel_deblurring_enhancer import KernelDeblurringEnhancer
from neural_de.utils.twe_logger import get_logger


def test_get_logger():
    test_log_file = "test_log.csv"
    try:
        os.remove(test_log_file)
    except FileNotFoundError:
        pass
    wmk = KernelDeblurringEnhancer(logger=get_logger(output= "both", filename= test_log_file))
    wmk._logger.debug("test")
    assert os.path.isfile(test_log_file)

