import hashlib
import os
from urllib import request
import warnings

from tqdm import tqdm


_PRETRAINED = {
    'PMC_CLIP': {
        'beta': 'https://huggingface.co/datasets/axiong/pmc_oa_beta/resolve/main/checkpoint.pt',
        'RC': 'https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/checkpoint.pt',
    }
}


def list_pretrained_tag_models(x):
    raise NotImplementedError('list_pretrained_tag_models not implemented')

def get_pretrained_url(model: str, tag: str):
    if model not in _PRETRAINED:
        raise RuntimeError("model not in _PRETRAINED")
    model_pretrained = _PRETRAINED[model]
    tag = tag.lower()
    if tag not in model_pretrained:
        raise RuntimeError(f'No macthed tag: {tag}')
    return model_pretrained[tag]


def download_pretrained(url: str, root: str = os.path.expanduser("./.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    if 'openaipublic' in url:
        expected_sha256 = url.split("/")[-2]
    else:
        expected_sha256 = ''

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            return download_target

    with request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target
