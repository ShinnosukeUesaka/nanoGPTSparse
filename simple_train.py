import argparse
import logging
import math
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

import wandb
from configs import MasterConfig, train_config_dict
from model import SparseGPT

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(config: MasterConfig) -> None:
    wandb.init(project=config.wandb_project, name=config.wandb_run_name, mode='online' if config.wandb_log else 'disabled', config=asdict(config))
    best_val_loss: float = 1e9

    model = SparseGPT(config.model_config)
    model.to(config.device)
    wandb.watch(model)

    optimizer = torch.optim.AdamW(model.get_param_groups_for_optimizer(config.weight_decay), lr=config.learning_rate, betas=(config.beta1, config.beta2), fused=(config.device == 'cuda'))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, partial(lr_lambda, warmup_iters=config.warmup_iters, lr_decay_iters=config.lr_decay_iters, learning_rate=config.learning_rate, min_lr=config.min_lr))

    for i in range(config.max_iters):
        x, y, mask = get_batch(config.dataset_dir, config.block_size, config.batch_size, 'train', config.device, load_mask=True)
        input_mask = torch.ones_like(mask, dtype=torch.bool) if config.use_only_a else mask

        optimizer.zero_grad()
        with torch.autocast(device_type=config.device, dtype=torch.bfloat16, enabled=(config.device == 'cuda')):
            logits = model(x, input_mask, y)
            loss, loss_a, loss_b = loss_fn(logits, y, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        # logging
        if i % config.log_interval == 0:
            wandb.log({"loss": loss.item(), "loss_a": loss_a.item(), "loss_b": loss_b.item(), "iter": i, "lr": scheduler.get_last_lr()[0]})
            print(f"iter {i}: loss {loss.item()}, loss_a {loss_a.item()}, loss_b {loss_b.item()}, lr {scheduler.get_last_lr()[0]}")

        # evaluation and checkpointing
        if i % config.eval_interval == 0:
            loss, loss_a, loss_b = eval_fn(model, config.eval_iters, config)
            wandb.log({"eval_loss": loss, "eval_loss_a": loss_a, "eval_loss_b": loss_b, "iter": i})
            if loss < best_val_loss:
                best_val_loss = loss
                torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, f"model_{i}.pt"))


def lr_lambda(it: int, warmup_iters: int, lr_decay_iters: int, learning_rate: float, min_lr: float) -> float:
    # 1) Linear warmup
    if it < warmup_iters:
        return (it + 1) / (warmup_iters + 1)
    # 2) Min LR after decay_iters
    if it > lr_decay_iters:
        return min_lr / learning_rate
    # 3) Cosine decay between warmup_iters and lr_decay_iters
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (min_lr / learning_rate) + coeff * (1 - (min_lr / learning_rate))

# PyTorch LambdaLR Scheduler




def get_batch(data_dir: str, block_size: int, batch_size: int, split: str, device: str, load_mask: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        mask = np.memmap(os.path.join(data_dir, 'train_mask.bin'), dtype=np.uint8, mode='r') if load_mask else None
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        mask = np.memmap(os.path.join(data_dir, 'val_mask.bin'), dtype=np.uint8, mode='r') if load_mask else None

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if load_mask:
        mask = torch.stack([torch.from_numpy((mask[i:i+block_size]).astype(np.bool_)) for i in ix])

    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    if load_mask:
        return x, y, mask
    else:
        return x, y, None

def loss_fn(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vocab_size = logits.shape[-1]
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.flatten(), ignore_index=-1, reduction='none')
    loss_a = loss[mask.flatten()].mean()
    loss_b = loss[~mask.flatten()].mean()
    loss = loss.mean()
    return loss, loss_a, loss_b

@torch.no_grad()
def eval_fn(model: SparseGPT, eval_iters: int, config: MasterConfig) -> Tuple[float, float, float]:
    model.eval()
    losses = torch.zeros(eval_iters)
    losses_a = torch.zeros(eval_iters)
    losses_b = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y, mask_a = get_batch(config.dataset_dir, config.block_size, config.batch_size, 'val', config.device, load_mask=True)
        input_mask = torch.ones_like(mask_a, dtype=torch.bool) if config.use_only_a else mask_a
        logits = model(idx=X, mask_a=input_mask, targets=Y)
        loss, loss_a, loss_b = loss_fn(logits, Y, mask_a)
        losses[k] = loss.item()
        losses_a[k] = loss_a.item()
        losses_b[k] = loss_b.item()
    model.train()
    return losses.mean().item(), losses_a.mean().item(), losses_b.mean().item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    config_name = parser.parse_args().config_name
    config_instance = train_config_dict[config_name]
    main(config_instance)