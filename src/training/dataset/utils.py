import logging
import random
import pandas as pd
import torch
import re
import jsonlines

def csv_loader(input_filename, img_key, caption_key, sep):
    r"""
    Load images, captions from Csv data
    """
    df = pd.read_csv(input_filename, sep=sep)
    images, captions = df[img_key].tolist(), df[caption_key].tolist()
    return images, captions

def jsonl_loader(input_filename, img_key, caption_key, sep):
    images, captions = [], []
    with jsonlines.open(input_filename) as reader:
        for obj in reader:
            images.append(obj[img_key])
            captions.append(obj[caption_key])
    return images, captions


def base_masker(caption, vocab, mask_token='<MASK>', pad_token='<PAD>', ratio=0.15, *rest):
    r"""Basic masking strategy, same as in BERT.
    Args:
        caption: str
        ratio: probability of token been masked out
    """
    tokenizer, = rest

    def measure_word_len(word):
        token_ids = tokenizer.encode(word)
        # tokens = [tokenizer.decode(x) for x in token_ids]
        return len(token_ids) - 2

    tokens = caption.split()
    bert_input_tokens = []
    output_mask = []
    bert_label_tokens = []  # 被 mask 的保留原词, 否则用 [PAD] 代替
    for i, token in enumerate(tokens):
        prob = random.random()
        if prob < ratio:
            prob /= ratio
            # 80% randomly change token to mask token
            if prob < 0.8:
                word_len = measure_word_len(token)
                bert_input_tokens += [mask_token] * word_len
            # 10% randomly change token to random token
            elif prob < 0.9:
                rand_token = random.choice(vocab).replace('</w>', '')
                word_len = measure_word_len(rand_token)
                # tokens[i] = random.randrange(self.tokenizer.vocab_size)
                bert_input_tokens += [rand_token]
            # 10% randomly change token to current token
            else:
                bert_input_tokens += [token]
                word_len = measure_word_len(token)
            output_mask += [1] * word_len
            bert_label_tokens += [token]
        else:
            word_len = measure_word_len(token)
            bert_input_tokens += [token]
            output_mask += [0] * word_len
            bert_label_tokens += [pad_token] * word_len

    logging.debug(f"\033[42moutput_mask:\033[0m {output_mask}")

    token_result = dict.fromkeys(["bert_input_tokens", "output_mask", "bert_label_tokens"], None)
    for key in token_result:
        token_result[key] = eval(key)  # HACK dark magic, could be dangerous
    return token_result


def encode_mlm(caption, vocab, mask_token: str, pad_token: str, ratio: float, tokenizer, args):
    r"""Process captions into masked input and ground truth
    Args:
        caption:
        vocab:
        mask_token:
        pad_token:
        ratio:
        tokenizer:
        args:
    Return:
        bert_input: 
        bert_label: 

    Reference Code:
    - [BERT-pytorch]()
    - [DataCollatorForWholeWordMask](https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L1072)
    """
    context_length = args.context_length
    masker = base_masker

    token_result = masker(
        caption,
        vocab,
        mask_token,
        pad_token,
        ratio,
        tokenizer,
    )  # Remove words for MLM task

    output_mask = token_result["output_mask"]
    output_mask += [0] * (context_length - len(output_mask))
    output_mask = torch.tensor(output_mask[:context_length])
    logging.debug(len(output_mask), output_mask)

    bert_input_tokens = token_result["bert_input_tokens"]
    bert_input = ' '.join(bert_input_tokens)
    bert_label_tokens = token_result["bert_label_tokens"]
    bert_label = ' '.join(bert_label_tokens)

    return bert_input, bert_label
