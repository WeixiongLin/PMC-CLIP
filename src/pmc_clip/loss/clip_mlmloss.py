import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import gather_features

class CLIP_MLMLoss(nn.Module):

    def __init__(self, args, cache_labels=False):
        super().__init__()
        self.local_loss = args.local_loss
        self.gather_with_grad = args.gather_with_grad
        self.cache_labels = cache_labels
        self.rank = args.rank
        self.world_size = args.world_size
        self.use_horovod = args.horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, prediction):
        image_features = prediction["image_features"]
        text_features = prediction["text_features"]
        logit_scale = prediction["logit_scale"]
        bert_prediction = prediction["bert_prediction"]

        device = image_features.device
        bert_label = prediction["bert_label"]

        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        # bert_prediction  # [128, 77, 49409]; bert_label  # [batch_size=128, n_ctx=77]
        # logging.debug(f"\033[42mbert_prediction\033[0m: {bert_prediction.shape}")  # [128, 49409, 77]

        # NOTE
        # bert_prediction([128, 30522, 77]); bert_label.shape(128, 77)
        # Q: 为什么 bert_prediction 的长度不是 77+2 而是 77, img_token 去哪里了;
        # A: 在 CLIP_MLM_IMG 中被 `x[:, :-2, :]` 截断了
        mlm_loss = F.nll_loss(bert_prediction.transpose(1, 2), bert_label, ignore_index=0)
        match_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return match_loss, mlm_loss
        # return 0.5 * match_loss +  0.5 * mlm_loss
