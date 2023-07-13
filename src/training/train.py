import json
import logging
import math
import os
import time
from contextlib import suppress

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from pmc_clip import ClipLoss, CLIP_MLMLoss
from .distributed import is_master


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def train_one_epoch_mlm(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    model.train()

    loss_class = {
        'PMC_CLIP': CLIP_MLMLoss,
    }[args.clip_model]
    loss = loss_class(args=args, cache_labels=True)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    total_loss_m = AverageMeter()
    match_loss_m = AverageMeter()
    mlm_loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            prediction = model(batch)
            logit_scale = prediction["logit_scale"]
            match_loss, mlm_loss = loss(prediction)
            # total_loss = loss(prediction)
            loss_weight = {
                "0": [0.5, 0.5],
                "5": [0.33, 0.66],
                "9": [1.0, 0],
                "10": [0.0, 1.0],
            }[args.loss_weight]
            total_loss = loss_weight[0] * match_loss + loss_weight[1] * mlm_loss

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(batch["images"])
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            total_loss_m.update(total_loss.item(), batch_size)
            match_loss_m.update(match_loss.item(), batch_size)
            mlm_loss_m.update(mlm_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Total Loss: {total_loss_m.val:#.5g} ({total_loss_m.avg:#.4g}) "
                f"Match Loss: {match_loss_m.val:#.5g} ({match_loss_m.avg:#.4g}) "
                f"MLM Loss: {mlm_loss_m.val:#.5g} ({mlm_loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "total_loss": total_loss_m.val,
                "match_loss": match_loss_m.val,
                "mlm_loss": mlm_loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def train_one_epoch(*args, **kwargs):
    raise NotImplementedError('train_one_epoch not implemented')
