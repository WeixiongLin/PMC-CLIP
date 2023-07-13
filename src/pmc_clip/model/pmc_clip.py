"""PMC-CLIP Model

Reference Code: [ViLT](https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/vision_transformer.py)
"""
from collections import OrderedDict
from dataclasses import dataclass
import logging
import math
from typing import Tuple, Union, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from pmc_clip.timm_model import TimmModel
from pmc_clip.utils import freeze_batch_norm_2d, to_2tuple

from transformers import AutoTokenizer, AutoModel

from .config import CLIPVisionCfg, CLIPTextCfg
from .blocks import Bottleneck, AttentionPool2d, ResNet, ModifiedResNet, LayerNorm, QuickGELU, ResidualAttentionBlock,\
                    Transformer


class PMC_CLIP(nn.Module):
    def __init__(
            self,
            args,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
    ):
        super().__init__()
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)

        self.args = args
        self.context_length = text_cfg.context_length
        self.device = torch.device(args.device)

        # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
        # memory efficient in recent PyTorch releases (>= 1.10).
        # NOTE: timm models always use native GELU regardless of quick_gelu flag.
        act_layer = QuickGELU if quick_gelu else nn.GELU

        if vision_cfg.timm_model_name:
            self.visual = TimmModel(
                vision_cfg.timm_model_name,
                pretrained=vision_cfg.timm_model_pretrained,
                pool=vision_cfg.timm_pool,
                proj=vision_cfg.timm_proj,
                embed_dim=embed_dim,
                image_size=vision_cfg.image_size
            )
            act_layer = nn.GELU  # so that text transformer doesn't use QuickGELU w/ timm models
        elif isinstance(vision_cfg.layers, (tuple, list)):
            VisualBackbone = {
                "RN50": ResNet,
                "ModifiedRN50": ModifiedResNet,
            }[vision_cfg.backbone]
            vision_heads = vision_cfg.width * 32 // vision_cfg.head_width

            self.visual = VisualBackbone(
                layers=vision_cfg.layers,
                output_dim=embed_dim,
                heads=vision_heads,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width
            )
        else:
            vision_heads = vision_cfg.width // vision_cfg.head_width
            self.visual = VisualTransformer(
                image_size=vision_cfg.image_size,
                patch_size=vision_cfg.patch_size,
                width=vision_cfg.width,
                layers=vision_cfg.layers,
                heads=vision_heads,
                mlp_ratio=vision_cfg.mlp_ratio,
                output_dim=embed_dim,
                act_layer=act_layer,
            )

        if text_cfg.bert_model_name:
            # Tokenizer
            tokenizer_name = text_cfg.bert_model_name
            assert text_cfg.bert_model_name == 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', \
                "Please check [CLS]'s token id"
            self.cls_id = 2  # [CLS]'s token id is 2, while it varies from tokenizers
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            self.text_encoder = AutoModel.from_pretrained(text_cfg.bert_model_name)#, return_dict=True)
        else:
            self.text_encoder = Transformer(
                width=text_cfg.width,
                layers=text_cfg.layers,
                heads=text_cfg.heads,
                act_layer=act_layer,
            )
        self.transformer_width = text_cfg.width
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, text_cfg.width))
        self.ln_final = LayerNorm(text_cfg.width)

        self.text_projection = nn.Parameter(torch.empty(text_cfg.width, embed_dim))
        self.mlm_projection = None
        if args.mlm:
            self.mlm_projection = nn.Parameter(torch.empty(text_cfg.width, text_cfg.vocab_size))
        self.softmax = nn.LogSoftmax(dim=-1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.img_special_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fusion_module = Transformer(
            width=text_cfg.width,
            layers=text_cfg.fusion_layers,
            heads=text_cfg.heads,
        )
        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        if hasattr(self.visual, 'init_parameters'):
            self.visual.init_parameters()

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer_width ** -0.5)
        if self.mlm_projection is not None:
            nn.init.normal_(self.mlm_projection, std=self.transformer_width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, batch, image_features):
        encoded_input = self.tokenizer(batch["bert_input"], padding='max_length', truncation=True, max_length=self.context_length, return_tensors='pt')
        encoded_label = self.tokenizer(batch['bert_label'], padding='max_length', truncation=True, max_length=self.context_length, return_tensors='pt')
        encoded_label = encoded_label['input_ids'].to(self.device)
        encoded_input['input_ids'] = encoded_input['input_ids'].to(self.device)  # [128, 77]

        x = self.text_encoder(
            input_ids = encoded_input['input_ids'],
            output_attentions = False
        )
        x = x['last_hidden_state']
        last_token_index = torch.nonzero( (encoded_input['input_ids'] == self.cls_id).squeeze() )
        text_features = x[torch.arange(x.shape[0]), last_token_index[:, 1]]
        text_features = text_features @ self.text_projection  # NOTE for matching
        
        # Fusion Module
        img = torch.unsqueeze(image_features, 1)  # [128, 1 ,768]
        B, _len, _dim = x.shape
        img_special_tokens = self.img_special_token.expand(B, -1, -1)  # [128, 1, embed_dim]
        x = torch.cat([x, img_special_tokens, img], dim=1)  # [128, 77+1+1, 768]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.fusion_module(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, :-2, :]  # Remove token [img_special_token, img]

        bert_prediction = None
        if self.args.mlm:
            bert_prediction = self.softmax(x @ self.mlm_projection)  # [batch_size=128, n_ctx=77, vocab_size=49409]

        attentions = None
        text_output = dict.fromkeys(["text_features", "bert_prediction", "attentions", "encoded_label"], None)
        for key in text_output:
            text_output[key] = eval(key)  # HACK dark magic, could be dangerous

        return text_output

    def forward(self, batch):
        image = batch["images"]
        image = image.to(device=self.device, non_blocking=True)

        if (image is None) or (batch["bert_input"] is None):
            raise RuntimeError('Missing Image OR Text in the input')

        image_features = self.encode_image(image)
        image_features = F.normalize(image_features['image_features'], dim=-1)  # [128, 768]

        text_output = self.encode_text(batch, image_features)
        text_features = F.normalize(text_output["text_features"], dim=-1)

        clip_prediction = dict.fromkeys(["image_features", "text_features", "logit_scale", "bert_label", "bert_prediction", "attentions"], None)
        clip_prediction.update({
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": self.logit_scale.exp(),
            "bert_label": text_output["encoded_label"],
            "bert_prediction": text_output["bert_prediction"],
            "attentions": text_output["attentions"]
        })
        return clip_prediction

