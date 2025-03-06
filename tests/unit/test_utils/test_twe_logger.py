from pytest import raises
from neural_de.utils._twe_logger import log_and_raise, get_logger
from unittest.mock import MagicMock


def test_log_and_raise():
    logger = get_logger()
    logger.error = MagicMock(return_value=None)
    with raises(TypeError):
        msg = "This was a mistake"
        log_and_raise(logger, TypeError, msg)
    logger.error.assert_called_once_with(msg)
