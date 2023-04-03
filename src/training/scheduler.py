'''
Learning rate scheduler
'''

import numpy as np


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps, **rest):
    min_lr = 0
    # min_lr = base_lr / 10.0
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = min_lr + _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = min_lr + 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def lr_calculator(base_lr, warmup_length, min_lr, steps, step):
    if step < warmup_length:
        lr = min_lr + _warmup_lr(base_lr, warmup_length, step)
    else:
        e = step - warmup_length
        es = steps - warmup_length
        lr = min_lr + 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
    return lr


def cosine_annealing_lr(optimizer, base_lr, warmup_length, steps, restarts):
    r"""
    Introduced in [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)

    Args:
        optimizer: Wrapped Optimizer
        base_lr: Maximum LR
        steps: Total Iters
        restarts: Restart Times during all Iters
    """
    cycle_len = steps // restarts

    def _lr_adjuster(step):
        step_cur = step % cycle_len  # step in current cycle
        cycle_idx = step // cycle_len  # Progress: [cycle_idx / restarts]
        decay = 0.1 ** cycle_idx
        cur_lr = base_lr * decay  # Max LR in current cycle

        lr = lr_calculator(
            base_lr=cur_lr,
            warmup_length=warmup_length,
            min_lr=0,
            steps=cycle_len,
            step=step_cur
        )
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster