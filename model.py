"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class SparseGPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd_a: int
    n_embd_b: int
    n_embd_attention: int
    dropout: float
    bias: bool
    rope_condense_ratio: float = 1
    rope_base: int = 500000



class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SparseCausalSelfAttention(nn.Module):

    def __init__(self, config: SparseGPTConfig):
        super().__init__()
        assert config.n_embd_attention % config.n_head == 0
        self.head_size = config.n_embd_attention // config.n_head
        # key, query, value projections for all heads, but in a batch
        self.c_attn_a = nn.Linear(config.n_embd_a, config.n_embd_attention * 3, bias=config.bias)
        self.c_attn_b = nn.Linear(config.n_embd_b, config.n_embd_attention * 3, bias=config.bias)
        # output projection
        self.c_proj_a = nn.Linear(config.n_embd_attention, config.n_embd_a, bias=config.bias)
        self.c_proj_b = nn.Linear(config.n_embd_attention, config.n_embd_b, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd_attention = config.n_embd_attention
        self.n_embd_a = config.n_embd_a
        self.n_embd_b = config.n_embd_b
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x_a, x_b, cos, sin, mask_a: torch.BoolTensor):
        x_a = self.c_attn_a(x_a)
        x_b = self.c_attn_b(x_b)
        kqv = interleave_tokens(x_a, x_b, mask_a)
        B, T, _ = kqv.size()
        kqv = kqv.view(B, T, -1, self.head_size)
        kqv = kqv.permute(0, 2, 1, 3) # (B, T, n_heads*3, head_size) -> (B, n_heads*3, T, head_size)
        q, k, v  = kqv.split(self.n_head, dim=1)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, -1) # re-assemble all head outputs side by side

        y_a = y[mask_a]
        y_b = y[~mask_a]

        # output projection
        y_a = self.resid_dropout(self.c_proj_a(y_a))
        y_b = self.resid_dropout(self.c_proj_b(y_b))
        return y_a, y_b

class MLP(nn.Module):

    def __init__(self, n_embd, bias, dropout):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LLaMAMLP(nn.Module):
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc_2 = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)

class SparseBlock(nn.Module):

    def __init__(self, config: SparseGPTConfig):
        super().__init__()
        self.ln_1_a = LayerNorm(config.n_embd_a, bias=config.bias)
        self.ln_1_b = LayerNorm(config.n_embd_b, bias=config.bias)
        self.attn = SparseCausalSelfAttention(config)
        self.ln_2_a = LayerNorm(config.n_embd_a, bias=config.bias)
        self.ln_2_b = LayerNorm(config.n_embd_b, bias=config.bias)
        self.ln_3_a = LayerNorm(config.n_embd_a, bias=config.bias)  
        self.ln_3_b = LayerNorm(config.n_embd_b, bias=config.bias)
        self.mlp_a = LLaMAMLP(config.n_embd_a)
        self.mlp_b = LLaMAMLP(config.n_embd_b)
        self.ln_4_a = LayerNorm(config.n_embd_a, bias=config.bias)
        self.ln_4_b = LayerNorm(config.n_embd_b, bias=config.bias)
    def forward(self, x_a, x_b, cos, sin, mask_a):
        atten_result_a, atten_result_b =  self.attn(self.ln_1_a(x_a), self.ln_1_b(x_b), cos, sin, mask_a)
        x_a = x_a + self.ln_2_a(atten_result_a)
        x_b = x_b + self.ln_2_b(atten_result_b)
        mlp_result_a = self.mlp_a(self.ln_3_a(x_a))
        mlp_result_b = self.mlp_b(self.ln_3_b(x_b))
        x_a = x_a + self.ln_4_a(mlp_result_a)
        x_b = x_b + self.ln_4_b(mlp_result_b)
        return x_a, x_b


def interleave_tokens(tokens_a: torch.Tensor, tokens_b: torch.Tensor, mask_a: torch.Tensor) -> torch.Tensor:
    B, T = mask_a.size()
    C = tokens_a.size(-1)

    interleaved = torch.zeros(B, T, C, device=tokens_a.device, dtype=tokens_a.dtype)
    flattend_text_mask = mask_a.view(-1)
    interleaved.view(-1, C)[flattend_text_mask] = tokens_a.view(-1, C)
    interleaved.view(-1, C)[~flattend_text_mask] = tokens_b.view(-1, C)
    interleaved = interleaved.contiguous()
    return interleaved
class SparseGPT(nn.Module):

    def __init__(self, config: SparseGPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.max_seq_length = config.block_size
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte_a = nn.Embedding(config.vocab_size, config.n_embd_a),
            wte_b = nn.Embedding(config.vocab_size, config.n_embd_b),
            wpe_a = nn.Embedding(config.block_size, config.n_embd_a),
            wpe_b = nn.Embedding(config.block_size, config.n_embd_b),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([SparseBlock(config) for _ in range(config.n_layer)]),
            ln_f_a = LayerNorm(config.n_embd_a, bias=config.bias),
            ln_f_b = LayerNorm(config.n_embd_b, bias=config.bias),
        ))
        self.lm_head_a = nn.Linear(config.n_embd_a, config.vocab_size, bias=False)
        self.lm_head_b = nn.Linear(config.n_embd_b, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte_a.weight = self.lm_head_a.weight # https://paperswithcode.com/method/weight-tying
        self.transformer.wte_b.weight = self.lm_head_b.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe_a.weight.numel()
            n_params -= self.transformer.wpe_b.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, mask_a: torch.Tensor, targets=None):
        assert mask_a.shape == idx.shape
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0).expand(B, T) # shape (B, T)
        
        cos = self.cos[:T]
        sin = self.sin[:T]

        # forward the GPT model itself
        idx_a = idx[mask_a]
        pos_a = pos[mask_a]
        idx_b = idx[~mask_a]
        pos_b = pos[~mask_a]
        tok_emb_a = self.transformer.wte_a(idx_a) # token embeddings of shape (b, t, n_embd)
        tok_emb_b = self.transformer.wte_b(idx_b) # token embeddings of shape (b, t, n_embd)
        pos_emb_a = self.transformer.wpe_a(pos_a) # position embeddings of shape (b, t, n_embd)
        pos_emb_b = self.transformer.wpe_b(pos_b) # position embeddings of shape (t, n_embd)
        x_a = self.transformer.drop(tok_emb_a + pos_emb_a)
        x_b = self.transformer.drop(tok_emb_b + pos_emb_b)
        for block in self.transformer.h:
            x_a, x_b = block(x_a, x_b, cos, sin, mask_a)
        x_a = self.transformer.ln_f_a(x_a)
        x_b = self.transformer.ln_f_b(x_b)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits_a = self.lm_head_a(x_a)
            logits_b = self.lm_head_b(x_b)
            logits = interleave_tokens(logits_a, logits_b, mask_a)
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.flatten(), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits_a = self.lm_head_a(x_a[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits_b = self.lm_head_b(x_b[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = interleave_tokens(logits_a, logits_b, mask_a)
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = SparseGPTConfig(**config_args)
        model = SparseGPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd_a//cfg.n_head, cfg.block_size #TODO make this more accurate
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}."
                             " This is likely because the input text exceeds the supported context length of this model.")
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
    
    def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.config.rope_adjustments is None:
            extra_config = None

        else:
            adjusted_params_required = ["factor", "low_freq_factor", "high_freq_factor", "original_max_seq_len"]
            params_present = [param in self.config.rope_adjustments for param in adjusted_params_required]
            num_params_present = sum(params_present)

            if num_params_present == 0:
                extra_config = None  # uses standard RoPE
            elif num_params_present == 4:
                # These parameters should always be used together so that we don't interfere with standard rope
                extra_config = {
                    "original_max_seq_len": self.config.rope_adjustments["original_max_seq_len"],
                    "factor": self.config.rope_adjustments["factor"],
                    "low_freq_factor": self.config.rope_adjustments["low_freq_factor"],
                    "high_freq_factor": self.config.rope_adjustments["high_freq_factor"],
                }
            else:
                # Some but not all parameters are specified; raise an error
                missing_params = [param for param, present in zip(adjusted_params_required, params_present) if not present]
                raise ValueError(
                    f"The following adjusted RoPE parameters are missing in rope_adjustments: {', '.join(missing_params)}. "
                    "All adjusted RoPE parameters must be specified together."
                )

        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.n_embd_attention // self.config.n_head, # head size
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
            extra_config=extra_config,
        )

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx



def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
    extra_config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced Transformer with Rotary Position Embedding.

    Args:
        seq_len (int): Sequence length.
        n_elem (int): Number of elements (head dimension).
        device (torch.device, optional): Device for tensor allocations.
        base (int, optional): Base for computing inverse frequencies.
        condense_ratio (int, optional): Ratio to condense the position indices.
        extra_config (dict, optional): Configuration parameters for frequency adjustments (used by Llama 3.1 and 3.2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine caches for RoPE.
    """

    # Compute the inverse frequencies theta
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    if extra_config is not None:
        orig_context_len = extra_config["original_max_seq_len"]
        factor = extra_config["factor"]
        low_freq_factor = extra_config["low_freq_factor"]
        high_freq_factor = extra_config["high_freq_factor"]

        wavelen = 2 * torch.pi / theta
        ratio = orig_context_len / wavelen
        smooth_factor = (ratio - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_factor = torch.clamp(smooth_factor, min=0.0, max=1.0)

        # Compute adjusted_theta without masked indexing
        adjusted_theta = (1 - smooth_factor) * (theta / factor) + smooth_factor * theta
        theta = adjusted_theta

    # Create position indices `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def batched_index_select(t, dim, idx):
    """index_select for batched index and unbatched t"""
    if idx.dim() == 1:
        return torch.index_select(t, dim, idx)

    *batch_shape, idx_size = idx.shape
    res = torch.index_select(t, dim, idx.reshape(-1))  # flat index
    # split out single batch idx
    res = res.view(*t.shape[:dim], -1, idx_size, *t.shape[dim + 1 :])
    # move batch dim to front, this is np.rollaxis(res, dim, 0) for tensors
    dims = [dim] + list(range(res.dim()))
    del dims[dim + 1]
    res = res.permute(dims)
    # unflatten batch dims
    res = res.view(*batch_shape, *res.shape[1:])
    return res


def batched_index_copy_(t, dim, idx, val):
    """Index copy for batched t, idx, val"""

    if t.device.type == "mps":
        # Normalize negative dimensions
        if dim < 0:
            dim = t.dim() + dim
        if idx.dim() == 1:
            idx_shape = [1] * val.dim()
            idx_shape[dim] = -1
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)
            t.scatter_(dim, idx_expanded, val)
            return t

        elif idx.dim() == 2:
            assert dim != 0, "Cannot index the batch dimension"
            batch_size = idx.size(0)
            idx_size = idx.size(1)
            assert batch_size == t.size(0) == val.size(0)

            idx_shape = [batch_size] + [1] * (val.dim() - 1)
            idx_shape[dim] = idx_size
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)

            t.scatter_(dim, idx_expanded, val)
            return t
        else:
            raise NotImplementedError(f"idx.dim() == {idx.dim()} not supported")

    else:
        if idx.dim() == 1:
            return t.index_copy_(dim, idx, val)

        assert idx.dim() == 2, f"multiple batch dims not yet {idx.shape=}"
        assert dim != 0, f"cannot index batch dim {dim=}"
        batch_size, idx_size = idx.shape
        assert batch_size == t.size(0)
        assert batch_size == val.size(0)

        # if we can view the batch and indexed dimensions together, we could
        # do index trickery. This is, sadly, not the case for kvcache so we
        # fall back to for loop
        for i in range(batch_size):
            unbatched_dim = dim if dim < 0 else dim - 1
            t[i].index_copy_(unbatched_dim, idx[i], val[i])
        return t


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    if cos.dim() > 1:
        # batch dimensions must align
        # sin/cos are (B, T, hs) so we unsqeeze -3 for nh
        # we count from back because all of apply_rope does
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)

