import dataclasses
from dataclasses import dataclass
from pathlib import Path

from model import SparseGPTConfig


@dataclass
class MasterConfig:
    # wandb logging
    model_config: SparseGPTConfig = dataclasses.field(default_factory=lambda: SparseGPTConfig(block_size=1024, vocab_size=6064, n_layer=12, n_head=12, n_embd_a=768, n_embd_b=768, n_embd_attention=768, bias=False))
    use_only_a: bool = False

    wandb_log: bool = False # disabled by default
    wandb_project: str = 'mini-gpt-sparse'
    wandb_run_name: str = 'gpt2' # 'run' + str(time.time())


    # data
    dataset_dir: Path = Path('data') / 'enwik8_char'

    batch_size: int = 32 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 1024

    learning_rate: float = 6e-4 # max learning rate
    max_iters: int = 600000 # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True # whether to decay the learning rate
    warmup_iters: int = 100 # how many steps to warm up for
    lr_decay_iters: int = 600000 # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    log_interval: int = 10
    eval_interval: int = 500
    eval_iters: int = 100
    checkpoint_dir: str = 'checkpoints'

    device: str = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    
train_config_dict: dict[str, MasterConfig] = {
    #"default": MasterConfig(),
    "mini": MasterConfig(model_config=SparseGPTConfig(n_layer=1, n_head=1, n_embd_a=64, n_embd_b=64, n_embd_attention=64, block_size=1024, vocab_size=6064, bias=False, dropout=0.0), max_iters=100000, device='mps', eval_iters=5, wandb_log=True, warmup_iters=100, lr_decay_iters=300),
    "sparse": MasterConfig(model_config=SparseGPTConfig(n_layer=9, n_head=8, n_embd_a=512, n_embd_b=512, n_embd_attention=512, block_size=2048, vocab_size=6064, bias=False, dropout=0.1), wandb_log=True, wandb_run_name="dense", use_only_a=False, checkpoint_dir="checkpoints_v2", lr_decay_iters=60000, max_iters=60000, block_size=2048, batch_size=16),
    "sparse_dropout": MasterConfig(model_config=SparseGPTConfig(n_layer=9, n_head=8, n_embd_a=512, n_embd_b=512, n_embd_attention=512, block_size=2048, vocab_size=6064, bias=False, dropout=0.2), wandb_log=True, wandb_run_name="sparse_dropout", use_only_a=False, checkpoint_dir="checkpoints_v3", lr_decay_iters=60000, max_iters=60000, block_size=2048, batch_size=16),
    "dense": MasterConfig(model_config=SparseGPTConfig(n_layer=9, n_head=8, n_embd_a=512, n_embd_b=512, n_embd_attention=512, block_size=2048, vocab_size=6064, bias=False, dropout=0.1), wandb_log=True, wandb_run_name="dense", use_only_a=True, checkpoint_dir="checkpoints_v4", lr_decay_iters=60000, max_iters=60000, block_size=2048, batch_size=16)
}




@dataclass
class EvaluateConfig:
    model_config: SparseGPTConfig
    dataset_dir: Path = Path('data') / 'enwik8_char'
    checkpoint_file: Path = Path('checkpoints') / 'checkpoint.pt'
    device: str = 'cuda'
    use_only_a: bool = True
    batch_size: int = 12
    eval_iters: int = 500

evaluate_config_dict: dict[str, EvaluateConfig] = {
    "sparse": EvaluateConfig(model_config=SparseGPTConfig(n_layer=9, n_head=8, n_embd_a=512, n_embd_b=512, n_embd_attention=512, block_size=2048, vocab_size=6064, bias=False, dropout=0.1), device='cuda', use_only_a=False, batch_size=16, eval_iters=500, checkpoint_file=Path('checkpoints_v2') / 'model_42500.pt'),
}