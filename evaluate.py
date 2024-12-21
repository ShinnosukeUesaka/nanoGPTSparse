import argparse
import logging
import math
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.nn import functional as F

import wandb
from configs import EvaluateConfig, evaluate_config_dict
from model import SparseGPT

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@torch.no_grad()
def main(config: EvaluateConfig):
    model = SparseGPT(config.model_config)
    model.load_state_dict(torch.load(config.checkpoint_file))
    model.to(config.device)
    model.eval()
    losses = torch.zeros(config.eval_iters)
    losses_a = torch.zeros(config.eval_iters)
    losses_b = torch.zeros(config.eval_iters)
    for k in tqdm.tqdm(range(config.eval_iters)):
        X, Y, mask_a = get_batch(config.dataset_dir, config.model_config.block_size, config.batch_size, 'test', config.device, load_mask=True)
        input_mask = torch.ones_like(mask_a, dtype=torch.bool) if config.use_only_a else mask_a
        logits = model(idx=X, mask_a=input_mask, targets=Y)
        loss, loss_a, loss_b = loss_fn(logits, Y, mask_a)
        losses[k] = loss.item()
        losses_a[k] = loss_a.item()
        losses_b[k] = loss_b.item()
    
    print(f"Loss: {losses.mean()}, Loss A: {losses_a.mean()}, Loss B: {losses_b.mean()}")



def get_batch(data_dir, block_size, batch_size, split, device, load_mask=False):
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
        return x, y

def loss_fn(logits, y, mask):
    vocab_size = logits.shape[-1]
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.flatten(), ignore_index=-1, reduction='none')
    loss_a = loss[mask.flatten()].mean()
    loss_b = loss[~mask.flatten()].mean()
    loss = loss.mean()
    return loss, loss_a, loss_b

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    config_name = parser.parse_args().config_name
    config_instance = evaluate_config_dict[config_name]
    main(config_instance)