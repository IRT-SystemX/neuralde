import logging

import torch as th
import torch.nn as nn

from neural_de.transformations.diffusion.unet.attention_block import AttentionBlock
from neural_de.transformations.diffusion.unet.downsample import Downsample
from neural_de.transformations.diffusion.unet.nn import (
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
from neural_de.transformations.diffusion.unet.res_block import ResBlock
from neural_de.transformations.diffusion.unet.timestep_embed_sequential import TimestepEmbedSequential
from neural_de.transformations.diffusion.unet.upsample import Upsample
from neural_de.transformations.diffusion.unet.utils import convert_module_to_f16, convert_module_to_f32
from neural_de.transformations.diffusion.diffpure_config import DiffPureConfig
from neural_de.utils.twe_logger import get_logger


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: DiffPureConfig,
        logger: logging.Logger = None
    ):
        super().__init__()

        self._logger = logger if logger is not None else get_logger()
        if config.num_heads_upsample == -1:
            config.num_heads_upsample = config.num_heads

        self._config = config
        self._dtype = th.float16 if self._config.use_fp16 else th.float32
        self._num_heads_upsample = config.num_heads_upsample
        # model_channels = num_channels
        time_embed_dim = self._config.num_channels * 4
        self.time_embed = nn.Sequential(
            linear(self._config.num_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self._config.num_classes is not None:
            self._label_emb = nn.Embedding(self._config.num_classes, time_embed_dim)

        ch = input_ch = int(self._config.channel_mult[0] * self._config.num_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(self._config.dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self._config.channel_mult):
            for _ in range(self._config.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        self._config.dropout,
                        out_channels=int(mult * self._config.num_channels),
                        dims=self._config.dims,
                        use_checkpoint=self._config.use_checkpoint,
                        use_scale_shift_norm=self._config.use_scale_shift_norm,
                    )
                ]
                ch = int(mult * self._config.num_channels)
                if ds in self._config.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self._config.use_checkpoint,
                            num_heads=self._config.num_heads,
                            num_head_channels=self._config.num_head_channels,
                            use_new_attention_order=self._config.use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self._config.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            self._config.dropout,
                            out_channels=out_ch,
                            dims=self._config.dims,
                            use_checkpoint=self._config.use_checkpoint,
                            use_scale_shift_norm=self._config.use_scale_shift_norm,
                            down=True,
                        )
                        if self._config.resblock_updown
                        else Downsample(
                            ch, self._config.conv_resample, dims=self._config.dims,
                            out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                self._config.dropout,
                dims=self._config.dims,
                use_checkpoint=self._config.use_checkpoint,
                use_scale_shift_norm=self._config.use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=self._config.use_checkpoint,
                num_heads=self._config.num_heads,
                num_head_channels=self._config.num_head_channels,
                use_new_attention_order=self._config.use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                self._config.dropout,
                dims=self._config.dims,
                use_checkpoint=self._config.use_checkpoint,
                use_scale_shift_norm=self._config.use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self._config.channel_mult))[::-1]:
            for i in range(self._config.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        self._config.dropout,
                        out_channels=int(self._config.num_channels * mult),
                        dims=self._config.dims,
                        use_checkpoint=self._config.use_checkpoint,
                        use_scale_shift_norm=self._config.use_scale_shift_norm,
                    )
                ]
                ch = int(self._config.num_channels * mult)
                if ds in self._config.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self._config.use_checkpoint,
                            num_heads=self._config.num_heads_upsample,
                            num_head_channels=self._config.num_head_channels,
                            use_new_attention_order=self._config.use_new_attention_order,
                        )
                    )
                if level and i == self._config.num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            self._config.dropout,
                            out_channels=out_ch,
                            dims=self._config.dims,
                            use_checkpoint=self._config.use_checkpoint,
                            use_scale_shift_norm=self._config.use_scale_shift_norm,
                            up=True,
                        )
                        if self._config.resblock_updown
                        else Upsample(ch, self._config.conv_resample, dims=self._config.dims,
                                      out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(self._config.dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self._config.num_channels))

        if self._config.num_classes is not None:
            emb = emb + self._label_emb(y)

        h = x.type(self._dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)
