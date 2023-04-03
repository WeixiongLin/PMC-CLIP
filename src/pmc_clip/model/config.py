from typing import Tuple, Union, Callable, Optional

from dataclasses import dataclass

@dataclass
class CLIPVisionCfg:
    backbone: str = 'ModifiedRN50'  # ['RN50', 'ModifiedRN50', 'MAE']
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    patch_dropout: float = 0.0  # patch dropout rate, no dropout by default
    drop_attention_rate: float = 0.  # Transformer Dropout


@dataclass
class CLIPTextCfg:
    bert_model_name: str = 'base'
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    fusion_layers: int = 1  # layers of fusion_module
    MOMENTUM: float = 0.5  # 0.99
