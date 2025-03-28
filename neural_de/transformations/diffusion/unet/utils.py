import torch.nn as nn


def convert_module_to_f16(ll):
    """
    Convert primitive modules to float16.
    """
    if isinstance(ll, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        ll.weight.data = ll.weight.data.half()
        if ll.bias is not None:
            ll.bias.data = ll.bias.data.half()


def convert_module_to_f32(ll):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(ll, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        ll.weight.data = ll.weight.data.float()
        if ll.bias is not None:
            ll.bias.data = ll.bias.data.float()
