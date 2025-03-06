import numpy as np
from neural_de.utils._validation import is_batch_valid, is_device_valid, is_power_of_two


def test_is_device_valid():
    assert is_device_valid("cpu")
    assert not is_device_valid("gpu")
    assert is_device_valid("cpu:0")
    assert is_device_valid("cuda")
    assert is_device_valid("cuda:0")
    assert not is_device_valid("ttpu")
    assert is_device_valid(0)


def test_is_batch_valid():
    assert not is_batch_valid([])[0]
    assert is_batch_valid(np.zeros((200, 4, 4, 1)))[0]
    assert is_batch_valid([np.ones((1280, 800, 3))])[0]
    assert is_batch_valid([np.ones((300, 300, 1)), np.ones((600, 400, 1))])[0]
    is_valid, msg = is_batch_valid([np.ones((300, 300, 1)), np.ones((600, 400, 1))], same_dim=True)
    assert not is_valid
    assert len(msg) > 0
    is_valid, msg = is_batch_valid([12])
    assert not is_valid
    assert len(msg) > 0


def test_is_power_of_two():
    assert not is_power_of_two(0)
    assert is_power_of_two(1)
    assert not is_power_of_two(3)
    assert not is_power_of_two(1000001)
    assert not is_power_of_two(20)
    assert is_power_of_two(1024)
