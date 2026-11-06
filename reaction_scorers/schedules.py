"""Learning-rate schedules for RP and RGP training."""

import numpy as np


def update_lr_for_step(optimizer, current_step, warmup_batches=10000):
    """RGP warmup sets the base learning rate before the plateau scheduler."""
    if current_step < warmup_batches:
        scale = min(1.0, float(current_step + 1) / warmup_batches)
        for group in optimizer.param_groups:
            group["lr"] = scale * optimizer.defaults["lr"]


def warmup_and_cosine_decay(
    step_num, warmup_steps=3000, total_steps=495944, decay_to=0.01
):
    """RP learning-rate multiplier: linear warmup, then cosine decay to 1%."""
    if step_num <= warmup_steps:
        return step_num / warmup_steps
    progress = (step_num - warmup_steps) / (total_steps - warmup_steps)
    return (1 - decay_to) * 0.5 * (1 + np.cos(progress * np.pi)) + decay_to
