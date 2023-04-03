import logging
import random
import pandas as pd
import jsonlines
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .utils import csv_loader, jsonl_loader, encode_mlm


class PmcDataset(Dataset):
    def __init__(self, args, input_filename, transforms, is_train):
        logging.debug(f'Loading csv data from {input_filename}.')

        self.args = args
        suffix = input_filename.split('.')[-1]
        loader = {
            'csv': csv_loader,
            'jsonl': jsonl_loader,
        }[suffix]
        self.images, self.captions = loader(
            input_filename=input_filename,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator
        )
        self.transforms = transforms

        if args.mlm:
            self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
            self.mask_token, self.pad_token = '[MASK]', '[PAD]'
            vocab = list(self.tokenizer.get_vocab().keys())
            # Remove special token from vocab
            self.vocab_with_no_special_token = [vocab_token for vocab_token in vocab if vocab_token not in self.tokenizer.all_special_tokens]
            self.ratio = args.mask_ratio if is_train else 0.0

        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        output = dict.fromkeys(["images", "bert_input", "bert_label"], None)
        if self.args.image_dir is None:
            image_path = self.images[idx]
        else:
            image_path = f'{self.args.image_dir}/{self.images[idx]}'
        images = self.transforms(Image.open(str(image_path)))
        caption = str(self.captions[idx])
        
        if self.args.mlm:  # MLM task
            bert_input, bert_label = encode_mlm(
                caption=caption,
                vocab=self.vocab_with_no_special_token,
                mask_token=self.mask_token,
                pad_token=self.pad_token,
                ratio=self.ratio,
                tokenizer=self.tokenizer,
                args=self.args,
            )
        else:
            bert_input = caption
            bert_label = caption  # FIXME pytorch forbid batch data to be none, forced to assign sth to bert_input/label

        output.update({
            "images": images,
            "bert_input": bert_input,
            "bert_label": bert_label
        })
        return output

