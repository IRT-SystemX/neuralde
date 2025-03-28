import os
from dataclasses import dataclass, field
from pathlib import Path

# where the weights will be download and store
ENHANCER = "diffpure"
MODEL_FILENAME = "256x256_diffusion_uncond.pt"
NEURALDE_MODEL_DOWNLOAD_PATH = Path(os.path.expanduser("~")) / ".neuralde" / ENHANCER / MODEL_FILENAME

# automatic channel_mult values depending on input image size
CHANNEL_MULT = dict([
    (512, (0.5, 1, 1, 2, 2, 4, 4)),
    (256, (1, 1, 2, 2, 4, 4)),
    (128, (1, 1, 2, 3, 4)),
    (64, (1, 2, 3, 4))])


@dataclass
class DiffPureConfig:
    """
    A dataclass to configure and provide parameters for the internal diffusion model of
    diffusion_enhancer.

    Most of the parameters are available to allow a custom usage of a different pre-trained
    diffusion models, based on the U-net architecture and code.
    The one which can be modified with the provided model are t, t_delta and sample_steps.

    Attributes:
        weights_path: Path of the pre-trained weights, to provide custom weights files.
        img_shape: the shape of each input image of the diffusion model (by default (3, 256, 256)).
         Dimension are hannel-first.
        attention_resolutions: resolution, in pixels, of the attention-layers of the model
        num_classes: int. (by default None). Number of classes the diffusion model is trained of.
        dims: int. images 1D, 2D or 3D (by default = 2)
        learn_sigma: bool (by default = True). If true, the output channel number will be 6 instead
         of 3.
        num_channels: int (by default 256). Base channel number for the layers of the diffusion
         model architecture.
        num_head_channels: int (by default 64). Number of channel per head of the attention blocks.
        num_res_blocks: int (by default 2). Number of residual block of the architecture.
        resblock_updown: bool (by default True). Whether to apply a downsampling after each residual
         block of the underlying Unet architecture.
        use_fp16: bool (by default True). Use 16bit floating -point precision. If cuda is not
         available, will be set as false (fp32).
        use_scale_shift_norm: bool (by default True). Normalisation of the output of each block
         of layers in the Unet architecture.
        num_heads: int (by default 4). Number of attention heads.
        num_heads_upsample: int (by default -1). Num head for upsampling attention layers.
        channel_mult: tuple (by default None). Will be computed if not provided. Depending on the
         resolution, multiply the base channel number to get the final one for each residual layer
         of the Unet model.
        dropout: float (by default 0.0). Dropout rate.
        use_new_attention_order: bool (by default False). If true, the unet will use QKVAttention
         layers, if False, will use QKVAttentionLegacy.
        t: int (by default 150). Number of diffusion steps applied for each image.
        t_delta: int (by default 15). Strength of the noise added before the diffusion process.
        use_bm: float (by default False) #Erreur sur la valeur?
        use_checkpoint: bool (by default False). gradient checkpointing for training
        conv_resample: bool (by default True). Use learned convolutions for upsampling and
         downsampling. If false, interpolation (nearest) will be used.
        sample_step: int (by default 1). Number of time the diffusion process (noise addition +
         denoising) is repeated for each image.
        rand_t: bool (by default False). If true, add random noise before denoising. The noise is
         sampled uniformly between -t_delta and +t_delta.
    """
    weights_path: Path = NEURALDE_MODEL_DOWNLOAD_PATH
    img_shape: tuple = (3, 256, 256)
    attention_resolutions: list[int] = field(default_factory=lambda: [32, 16, 8])
    num_classes: int = None
    dims: int = 2  # 1D, 2D or 3D images
    learn_sigma: bool = True
    num_channels: int = 256
    num_head_channels: int = 64
    num_res_blocks: int = 2
    resblock_updown: bool = True
    use_fp16: bool = True
    use_scale_shift_norm: bool = True
    num_heads: int = 4
    num_heads_upsample: int = -1
    channel_mult: tuple = None
    dropout: float = 0.0
    use_new_attention_order: bool = False
    t: int = 150
    t_delta: int = 15
    use_bm: float = False
    use_checkpoint: bool = False
    conv_resample: bool = True
    sample_step: int = 1
    rand_t: bool = False

    def __post_init__(self):
        """
        Post-init parameters inference to avoid redundant parameters
        """
        self.image_size: int = self.img_shape[-1]
