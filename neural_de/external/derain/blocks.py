"""
2D convolution class
"""
import functools

import torch
import torchvision
from torch import nn


class Conv2d(torch.nn.Module):
    """
    2D convolution class
    Args:
    in_channels : int - Number of input channels
    out_channels : int - Number of output channels
    kernel_size : int - Size of kernel
    stride : int - Stride of convolution
    activation_func : func - Activation function after convolution
    norm_layer : functools.partial - Normalization layer
    use_bias : bool - If set, then use bias
    padding_type : str - The name of padding layer: reflect | replicate | zero
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
            norm_layer=nn.BatchNorm2d,
            use_bias=False,
            padding_type="reflect",
    ):
        super(Conv2d, self).__init__()

        self.activation_func = activation_func
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(kernel_size // 2)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(kernel_size // 2)]
        elif padding_type == "zero":
            p = kernel_size // 2
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(
                in_channels,
                out_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=p,
                bias=use_bias,
            ),
            norm_layer(out_channels),
        ]

        self.conv = nn.Sequential(*conv_block)

    def forward(self, x):
        conv = self.conv(x)

        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class DeformableConv2d(nn.Module):
    """
    2D deformable convolution class
    Args:
        in_channels : int - number of input channels
        out_channels : int - number of output channels
        kernel_size : int - size of kernel
        stride : int - stride of convolution
        padding : int - padding
        use_bias : bool - if set, then use bias
    """

    def __init__(
            self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    ):
        super(DeformableConv2d, self).__init__()

        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True,
        )

        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        self.modulator_conv = nn.Conv2d(
            in_channels,
            1 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True,
        )

        nn.init.constant_(self.modulator_conv.weight, 0.0)
        nn.init.constant_(self.modulator_conv.bias, 0.0)

        self.regular_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
        )
        return x


class UpConv2d(torch.nn.Module):
    """
    Up-convolution (upsample + convolution) block class
    Args:
    in_channels : int - number of input channels
    out_channels : int - number of output channels
    kernel_size : int - size of kernel (k x k)
    activation_func : func - activation function after convolution
    norm_layer : functools.partial - normalization layer
    use_bias : bool - if set, then use bias
    padding_type : str - the name of padding layer: reflect | replicate | zero
    interpolate_mode : str - the mode for interpolation: bilinear | nearest
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
            norm_layer=nn.BatchNorm2d,
            use_bias=False,
            padding_type="reflect",
            interpolate_mode="bilinear",
    ):
        super(UpConv2d, self).__init__()
        self.interpolate_mode = interpolate_mode

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            activation_func=activation_func,
            norm_layer=norm_layer,
            use_bias=use_bias,
            padding_type=padding_type,
        )

    def forward(self, x):
        n_height, n_width = x.shape[2:4]
        shape = (int(2 * n_height), int(2 * n_width))
        upsample = torch.nn.functional.interpolate(
            x, size=shape, mode=self.interpolate_mode, align_corners=True
        )
        conv = self.conv(upsample)
        return conv


class DeformableResnetBlock(nn.Module):
    """Define a Resnet block with deformable convolutions"""

    def __init__(
            self, dim, padding_type, norm_layer, use_dropout, use_bias, activation_func
    ):
        """
        Initialize the deformable Resnet block
        A deformable resnet block is a conv block with skip connections
        """
        super(DeformableResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias, activation_func
        )

    def build_conv_block(
            self, dim, padding_type, norm_layer, use_dropout, use_bias, activation_func
    ):
        """
        Construct a convolutional block.
        Parameters:
            dim (int) -- the number of channels in the conv layer.
            padding_type (str) -- the name of padding layer: reflect | replicate | zero
            norm_layer -- normalization layer
            use_dropout (bool) -- if use dropout layers.
            use_bias (bool) -- if the conv layer uses bias or not
            activation_func (func) -- activation type
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer)
        """
        conv_block = []

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            DeformableConv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            activation_func,
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            DeformableConv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class DecoderBlock(torch.nn.Module):
    """
    Decoder block with skip connections
    Args:
    in_channels : int - number of input channels
    skip_channels : int - number of skip connection channels
    out_channels : int - number of output channels
    activation_func : func - activation function after convolution
    norm_layer : functools.partial - normalization layer
    use_bias : bool - if set, then use bias
    padding_type : str - the name of padding layer: reflect | replicate | zero
    upsample_mode : str - the mode for interpolation: transpose | bilinear | nearest
    """

    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
            norm_layer=nn.BatchNorm2d,
            use_bias=False,
            padding_type="reflect",
            upsample_mode="transpose",
    ):
        super(DecoderBlock, self).__init__()

        self.skip_channels = skip_channels
        self.upsample_mode = upsample_mode

        # Upsampling
        if upsample_mode == "transpose":
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(out_channels),
                activation_func,
            )
        else:
            self.deconv = UpConv2d(
                in_channels,
                out_channels,
                use_bias=use_bias,
                activation_func=activation_func,
                norm_layer=norm_layer,
                padding_type=padding_type,
                interpolate_mode=upsample_mode,
            )

        concat_channels = skip_channels + out_channels

        self.conv = Conv2d(
            concat_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            activation_func=activation_func,
            padding_type=padding_type,
            norm_layer=norm_layer,
            use_bias=use_bias,
        )

    def forward(self, x, skip=None):
        deconv = self.deconv(x)

        if self.skip_channels > 0:
            concat = torch.cat([deconv, skip], dim=1)
        else:
            concat = deconv

        return self.conv(concat)


class ResNetModified(nn.Module):
    """
    Resnet-based generator that consists of deformable Resnet blocks.
    """

    def __init__(
            self,
            input_nc,
            output_nc,
            ngf=64,
            norm_layer=nn.BatchNorm2d,
            activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
            use_dropout=False,
            n_blocks=6,
            padding_type="reflect",
            upsample_mode="bilinear",
    ):
        """
        Construct a Resnet-based generator
        Parameters:
          input_nc (int) -- the number of channels in input images
          output_nc (int) -- the number of channels in output images
          ngf (int) -- the number of filters in the last conv layer
          norm_layer -- normalization layer
          use_dropout (bool) -- if use dropout layers
          n_blocks (int) -- the number of ResNet blocks
          padding_type (str) -- the name of padding layer in conv layers: reflect | replicate | zero
          upsample_mode (str) -- mode for upsampling: transpose | bilinear
        """
        assert n_blocks >= 0
        super(ResNetModified, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Initial Convolution
        self.initial_conv = nn.Sequential(
            Conv2d(
                in_channels=input_nc,
                out_channels=ngf,
                kernel_size=7,
                padding_type=padding_type,
                norm_layer=norm_layer,
                activation_func=activation_func,
                use_bias=use_bias,
            ),
            Conv2d(
                in_channels=ngf,
                out_channels=ngf,
                kernel_size=3,
                padding_type=padding_type,
                norm_layer=norm_layer,
                activation_func=activation_func,
                use_bias=use_bias,
            ),
        )

        # Downsample Blocks
        n_downsampling = 2
        mult = 2 ** 0
        self.downsample_1 = Conv2d(
            in_channels=ngf * mult,
            out_channels=ngf * mult * 2,
            kernel_size=3,
            stride=2,
            padding_type=padding_type,
            norm_layer=norm_layer,
            activation_func=activation_func,
            use_bias=use_bias,
        )

        mult = 2 ** 1
        self.downsample_2 = Conv2d(
            in_channels=ngf * mult,
            out_channels=ngf * mult * 2,
            kernel_size=3,
            stride=2,
            padding_type=padding_type,
            norm_layer=norm_layer,
            activation_func=activation_func,
            use_bias=use_bias,
        )

        # Residual Blocks
        residual_blocks = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            residual_blocks += [
                DeformableResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                    activation_func=activation_func,
                )
            ]

        self.residual_blocks = nn.Sequential(*residual_blocks)

        # Upsampling
        mult = 2 ** (n_downsampling - 0)
        self.upsample_2 = DecoderBlock(
            ngf * mult,
            int(ngf * mult / 2),
            int(ngf * mult / 2),
            use_bias=use_bias,
            activation_func=activation_func,
            norm_layer=norm_layer,
            padding_type=padding_type,
            upsample_mode=upsample_mode,
        )

        mult = 2 ** (n_downsampling - 1)
        self.upsample_1 = DecoderBlock(
            ngf * mult,
            int(ngf * mult / 2),
            int(ngf * mult / 2),
            use_bias=use_bias,
            activation_func=activation_func,
            norm_layer=norm_layer,
            padding_type=padding_type,
            upsample_mode=upsample_mode,
        )

        # Output Convolution
        self.output_conv_naive = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0),
            nn.Tanh(),
        )

    def forward(self, input):
        """Standard forward"""

        # Downsample
        initial_conv_out = self.initial_conv(input)
        downsample_1_out = self.downsample_1(initial_conv_out)
        downsample_2_out = self.downsample_2(downsample_1_out)

        # Residual
        residual_blocks_out = self.residual_blocks(downsample_2_out)

        # Upsample
        upsample_2_out = self.upsample_2(residual_blocks_out, downsample_1_out)
        upsample_1_out = self.upsample_1(upsample_2_out, initial_conv_out)
        final_out = self.output_conv_naive(upsample_1_out)

        return (final_out,)
