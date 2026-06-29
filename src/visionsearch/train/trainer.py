"""Training loop: AdamW over heads + temperature, AMP, gradient accumulation."""
from __future__ import annotations

from typing import Iterable

import torch
from torch.amp import GradScaler, autocast

from ..models.dual_encoder import DualEncoder
from .loss import ContrastiveLoss


def fit(
    model: DualEncoder,
    loss_fn: ContrastiveLoss,
    batches: Iterable[dict],
    *,
    epochs: int = 1,
    lr: float = 1e-3,
    device: str = "cpu",
    accum_steps: int = 1,
    use_amp: bool = True,
    log_dir: str | None = None,
    max_steps: int | None = None,
    on_epoch_end=None,
) -> list[float]:
    model.to(device)
    loss_fn.to(device)
    model.train()

    params = list(model.trainable_parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)

    amp_on = use_amp and device == "cuda"
    scaler = GradScaler(device, enabled=amp_on)

    writer = None
    if log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir)

    history: list[float] = []
    step = 0
    optimizer.zero_grad()
    for epoch in range(epochs):
        for i, batch in enumerate(batches):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device_type=device, enabled=amp_on):
                image_embeds, text_embeds = model(batch)
                loss = loss_fn(image_embeds, text_embeds)
            scaler.scale(loss / accum_steps).backward()
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            history.append(loss.item())
            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), step)
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        if on_epoch_end is not None:
            on_epoch_end(epoch)
        if max_steps is not None and step >= max_steps:
            break

    if writer is not None:
        writer.close()
    return history
