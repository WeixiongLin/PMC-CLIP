import json
import logging
import os
from contextlib import suppress

import numpy as np
import torch
import torch.nn.functional as F

from .distributed import is_master


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        cumulative_total_loss = 0.0
        cumulative_match_loss = 0.0
        cumulative_mlm_loss = 0.0
        cumulative_mim_loss = 0.0

        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                with autocast():
                    prediction = model(batch)

                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(prediction["image_features"].cpu())
                    all_text_features.append(prediction["text_features"].cpu())
                    logit_scale = prediction["logit_scale"].mean()
                    logits_per_image = logit_scale * prediction["image_features"] @ prediction["text_features"].t()
                    logits_per_text = logits_per_image.t()

                    # batch_size = args.batch_size  # NOTE the last batch in epoch is not full, could cause a crush
                    batch_size = len(batch["images"])
                    labels = torch.arange(batch_size, device=device).long()

                    match_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2
                    if args.mlm:
                        bert_prediction = prediction["bert_prediction"].transpose(1, 2)
                        mlm_loss = F.nll_loss(bert_prediction, prediction["bert_label"], ignore_index=0)
                    else:
                        mlm_loss = match_loss

                    mim_loss = prediction['mim_loss'] if args.mim else match_loss
                    total_loss = 0.33 * match_loss + 0.33 * mlm_loss + 0.33 * mim_loss

                cumulative_total_loss += total_loss * batch_size
                cumulative_match_loss += match_loss * batch_size
                cumulative_mlm_loss += mlm_loss * batch_size
                cumulative_mim_loss += mim_loss * batch_size
                num_samples += batch_size

                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_total_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            metrics.update({
                **val_metrics,
                "val_loss": (cumulative_total_loss / num_samples).item(),
                "match_loss": (cumulative_match_loss / num_samples).item(),
                "mlm_loss": (cumulative_mlm_loss / num_samples).item(),
                "mim_loss": (cumulative_mim_loss / num_samples).item(),
                "epoch": epoch,
                "num_samples": num_samples
            })

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    # image_features: [data_size, 768]
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    # logits = {"image_to_text": logits_per_image}
    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


