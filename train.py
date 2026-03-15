"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import csv
import gc
import json
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def report_environment():
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    print(f"Environment detected: {device_name}")
    print()

report_environment()

from prepare import (
    MAX_SEQ_LEN,
    TIME_BUDGET,
    Tokenizer,
    evaluate_policy,
    get_experiment_manifest,
    make_dataloader,
)

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def create_run_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(value) for value in obj]
    return obj


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def git_metadata() -> dict[str, object]:
    repo_dir = Path(__file__).resolve().parent

    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "is_dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }

# ---------------------------------------------------------------------------
# Fast value MLP for search guidance
# ---------------------------------------------------------------------------

class ValueMLP(nn.Module):
    """Tiny MLP that predicts distance-to-goal from raw sticker colors.
    ~100K params, evaluates in microseconds — fast enough for search."""

    def __init__(self, n_stickers: int = 24, n_colors: int = 6, hidden: int = 256):
        super().__init__()
        self.embed = nn.Embedding(n_colors, 16)
        self.net = nn.Sequential(
            nn.Linear(n_stickers * 16, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, stickers: torch.Tensor) -> torch.Tensor:
        """stickers: (B, 24) integer tensor of color indices 0-5."""
        x = self.embed(stickers)       # (B, 24, 16)
        x = x.view(x.size(0), -1)     # (B, 384)
        return self.net(x).squeeze(-1) # (B,)

# Color token ID → color index mapping (built at runtime)
_COLOR_TOKEN_MAP: dict[int, int] | None = None

def _get_color_map(tokenizer) -> dict[int, int]:
    global _COLOR_TOKEN_MAP
    if _COLOR_TOKEN_MAP is None:
        _COLOR_TOKEN_MAP = {
            tokenizer.token_to_id[f"COL_{c}"]: i
            for i, c in enumerate(("W", "Y", "G", "B", "R", "O"))
        }
    return _COLOR_TOKEN_MAP

def _extract_stickers_from_batch(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """Extract 24 sticker color indices from input_ids batch. Stickers at positions 5-28."""
    color_map = _get_color_map(tokenizer)
    sticker_tokens = input_ids[:, 5:29]  # (B, 24) COL_X token IDs
    stickers = torch.zeros_like(sticker_tokens)
    for token_id, color_idx in color_map.items():
        stickers[sticker_tokens == token_id] = color_idx
    return stickers

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # PyTorch SDPA without FlashAttention 3
        # Expand heads for KV based on GQA
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        
        # Transpose to [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply mask for sliding window
        window = window_size[0]
        if window > 0 and window < T:
            # Mask out tokens outside the window
            mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
            mask = mask.triu(diagonal=1 - window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Value head: predicts distance-to-goal (scalar) for search guidance
        self.value_head = nn.Linear(config.n_embd, 1, bias=True)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Value head: init to predict ~5 (midrange distance)
        torch.nn.init.normal_(self.value_head.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.value_head.bias, 5.0)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        value_head_params = list(self.value_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(value_head_params) +
            len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _backbone(self, idx):
        """Run transformer backbone, return final hidden states."""
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        return norm(x)

    def forward(self, idx, targets=None, distances=None, reduction='mean'):
        x = self._backbone(idx)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            policy_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                          ignore_index=-1, reduction=reduction)
            if distances is not None:
                # Value prediction at the position before the supervised token
                # Use mask multiplication to avoid gather (MPS bf16 backward compat)
                sup_mask = (targets != -1).float()  # (B, T) — 1 at supervised position
                # Shift mask left by 1 to get the position BEFORE the answer token
                value_mask = torch.zeros_like(sup_mask)
                value_mask[:, :-1] = sup_mask[:, 1:]
                # value_mask has a 1 at the last prompt position for each example
                x_f32 = x.float()
                value_all = self.value_head(x_f32).squeeze(-1)  # (B, T)
                value_pred = (value_all * value_mask).sum(dim=1)  # (B,)
                value_loss = F.mse_loss(value_pred, distances)
                return policy_loss + 0.5 * value_loss  # auxiliary value objective
            return policy_loss
        return logits

    def predict_value(self, idx):
        """Predict distance-to-goal from state tokens. Used for search."""
        x = self._backbone(idx)
        # Use the last non-pad position for value prediction
        value_pred = self.value_head(x[:, -1, :]).squeeze(-1)
        return value_pred

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    # Move scalars to correct device and dtype
    step_t = step_t.to(device=p.device, dtype=p.dtype)
    lr_t = lr_t.to(device=p.device, dtype=p.dtype)
    beta1_t = beta1_t.to(device=p.device, dtype=p.dtype)
    beta2_t = beta2_t.to(device=p.device, dtype=p.dtype)
    eps_t = eps_t.to(device=p.device, dtype=p.dtype)
    wd_t = wd_t.to(device=p.device, dtype=p.dtype)
    
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Move scalars to correct device and dtype
    momentum_t = momentum_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    lr_t = lr_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    wd_t = wd_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    beta2_t = beta2_t.to(device=stacked_params.device, dtype=stacked_params.dtype)

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    
    # Needs to match second_momentum_buffer.dtype for lerp_
    beta2_cast = beta2_t.to(second_momentum_buffer.dtype)
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2_cast)
    
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        
        # Compile conditionally
        compiler_kwargs = {"dynamic": False, "fullgraph": True}
        if device_type in ("cuda", "cpu"):
            self.adamw_step_fused = torch.compile(adamw_step_fused, **compiler_kwargs)
            self.muon_step_fused = torch.compile(muon_step_fused, **compiler_kwargs)
        else:
            self.adamw_step_fused = adamw_step_fused
            self.muon_step_fused = muon_step_fused

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            self.adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        self.muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 64           # target head dimension for attention (4 heads for more diverse patterns)
WINDOW_PATTERN = "L"    # sliding window pattern: L=full, S=half context

# Sequence length for training (flat state + up to 3 history moves + 1-token answer ≈ 34 tokens)
TRAIN_SEQ_LEN = 36

# Optimization
TOTAL_BATCH_SIZE = 18432  # = 512 * 36, one microstep per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.12        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.0      # no weight decay (best config)
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.05     # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.15   # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 512  # per-device batch size (reduce if OOM)

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")

# Detect device
device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_type)

# Autocast context
if device_type == "cuda":
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
elif device_type == "cpu":
    autocast_ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
else:
    import contextlib
    autocast_ctx = contextlib.nullcontext()

H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

run_dir = create_run_dir()
metrics_csv_path = run_dir / "train_metrics.csv"
summary_json_path = run_dir / "summary.json"
loss_plot_path = run_dir / "loss.png"
print(f"Run directory: {run_dir}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

summary_base = {
    "status": "running",
    "started_at": now_iso(),
    "run_dir": str(run_dir),
    "git": git_metadata(),
    "config": {
        "device_type": device_type,
        "torch_seed": 42,
        "max_seq_len": MAX_SEQ_LEN,
        "train_seq_len": TRAIN_SEQ_LEN,
        "depth": DEPTH,
        "device_batch_size": DEVICE_BATCH_SIZE,
        "total_batch_size": TOTAL_BATCH_SIZE,
        "aspect_ratio": ASPECT_RATIO,
        "head_dim": HEAD_DIM,
        "window_pattern": WINDOW_PATTERN,
        "embedding_lr": EMBEDDING_LR,
        "unembedding_lr": UNEMBEDDING_LR,
        "matrix_lr": MATRIX_LR,
        "scalar_lr": SCALAR_LR,
        "weight_decay": WEIGHT_DECAY,
        "adam_betas": ADAM_BETAS,
        "warmup_ratio": WARMUP_RATIO,
        "warmdown_ratio": WARMDOWN_RATIO,
        "final_lr_frac": FINAL_LR_FRAC,
    },
    "model_config": make_json_safe(asdict(config)),
    "protocol": make_json_safe(get_experiment_manifest()),
    "artifacts": {
        "metrics_csv": str(metrics_csv_path),
        "loss_plot": str(loss_plot_path),
        "summary_json": str(summary_json_path),
    },
}
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump(summary_base, f, indent=2)

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * TRAIN_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
summary_base["config"]["grad_accum_steps"] = grad_accum_steps
summary_base["config"]["num_params_m"] = num_params / 1e6
summary_base["config"]["estimated_flops_per_token"] = num_flops_per_token
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump(summary_base, f, indent=2)

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

# torch.compile is unstable on MPS, only use on CUDA
if device_type == "cuda":
    model = torch.compile(model, dynamic=False)

# Value MLP for fast search guidance
value_mlp = ValueMLP().to(device)
value_mlp_optimizer = torch.optim.Adam(value_mlp.parameters(), lr=1e-3)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, TRAIN_SEQ_LEN, "train")
x, y, d, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

metrics_file = open(metrics_csv_path, "w", newline="", encoding="utf-8")
metrics_writer = csv.DictWriter(
    metrics_file,
    fieldnames=[
        "step",
        "progress",
        "loss",
        "lr_multiplier",
        "dt_ms",
        "tok_per_sec",
        "mfu_percent",
        "epoch",
        "remaining_seconds",
    ],
)
metrics_writer.writeheader()
metrics_file.flush()
loss_history: list[tuple[int, float]] = []

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# DAgger: mid-training on-policy data collection
# ---------------------------------------------------------------------------

DAGGER_TRIGGER_FRACS = [0.5]  # single DAgger round at midpoint
DAGGER_NUM_EPISODES = 200   # rollouts to collect
DAGGER_MAX_STEPS = 15       # steps per rollout

from prepare import (
    _select_move_with_search,
    encode_supervised_example,
    _example_stream,
    get_runtime_device,
)
from rubiks import Cube, build_prompt_tokens, build_answer_tokens, random_scramble, scramble_length_for_size
from teacher_dwalton import solve_cube_222
import random as _random


def collect_dagger_data(model, tokenizer, num_episodes=DAGGER_NUM_EPISODES,
                        max_steps=DAGGER_MAX_STEPS):
    """Roll out the current policy and collect teacher corrections at visited states."""
    rng = _random.Random(int(time.time()))  # different data each run
    examples = []
    for _ in range(num_episodes):
        scramble = random_scramble(size=2, length=scramble_length_for_size(2), rng=rng)
        cube = Cube(2)
        cube.apply_moves(scramble)
        history = []
        visited = {cube.to_kociemba_string()}

        for _ in range(max_steps):
            if cube.has_uniform_faces():
                break

            # Get teacher correction at this on-policy state
            try:
                teacher_solution = solve_cube_222(cube)
                if teacher_solution:
                    prompt_tokens = build_prompt_tokens(2, cube, history=history)
                    answer_tokens = build_answer_tokens(teacher_solution[0])
                    encoded = encode_supervised_example(tokenizer, prompt_tokens, answer_tokens)
                    encoded["size"] = 2
                    encoded["distance_to_goal"] = len(teacher_solution)
                    examples.append(encoded)
            except Exception:
                pass

            # Follow the MODEL's policy (not teacher's) to visit on-policy states
            move = _select_move_with_search(model, tokenizer, cube, history, visited)
            if move is None:
                break
            try:
                cube.apply_move(move)
            except Exception:
                break
            history.append(move)
            visited.add(cube.to_kociemba_string())

    return examples


def _make_dataloader_from_examples(tokenizer, examples, B, T, shuffle=True):
    """Create a dataloader from an in-memory list of examples."""
    stream = _example_stream(examples, shuffle=shuffle)
    device_str = get_runtime_device()
    pad_id = tokenizer.get_pad_token_id()

    cpu_inputs = torch.full((B, T), pad_id, dtype=torch.long, pin_memory=(device_str == "cuda"))
    cpu_targets = torch.full((B, T), -1, dtype=torch.long, pin_memory=(device_str == "cuda"))
    cpu_distances = torch.zeros(B, dtype=torch.float32, pin_memory=(device_str == "cuda"))
    inputs = torch.full((B, T), pad_id, dtype=torch.long, device=device_str)
    targets = torch.full((B, T), -1, dtype=torch.long, device=device_str)
    distances = torch.zeros(B, dtype=torch.float32, device=device_str)

    while True:
        for row_idx in range(B):
            example, epoch = next(stream)
            input_ids = example["input_ids"]
            target_ids = example["targets"]
            seq_len = min(len(input_ids), T)
            cpu_inputs[row_idx].fill_(pad_id)
            cpu_targets[row_idx].fill_(-1)
            cpu_inputs[row_idx, :seq_len] = torch.tensor(input_ids[:seq_len], dtype=torch.long)
            cpu_targets[row_idx, :seq_len] = torch.tensor(target_ids[:seq_len], dtype=torch.long)
            cpu_distances[row_idx] = float(example.get("distance_to_goal", 0))

        inputs.copy_(cpu_inputs, non_blocking=(device_str == "cuda"))
        targets.copy_(cpu_targets, non_blocking=(device_str == "cuda"))
        distances.copy_(cpu_distances, non_blocking=(device_str == "cuda"))
        yield inputs, targets, distances, epoch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
dagger_round = 0  # which DAgger round we're on (0 = not yet triggered)
all_dagger_examples: list = []

def sync_device(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()

while True:
    sync_device(device_type)
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y, distances=d)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, d, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # Train value MLP on same batch (fast, negligible overhead)
    stickers = _extract_stickers_from_batch(x, tokenizer)
    mlp_pred = value_mlp(stickers)
    mlp_loss = F.mse_loss(mlp_pred, d)
    mlp_loss.backward()
    value_mlp_optimizer.step()
    value_mlp_optimizer.zero_grad()

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding
    if train_loss_f > 100:
        print("FAIL")
        exit(1)

    sync_device(device_type)
    t1 = time.time()
    dt = t1 - t0

    if step > 10 and dt < 5.0:  # exclude MPS stalls from time budget
        total_training_time += dt

    # DAgger: collect on-policy data at each trigger fraction
    if dagger_round < len(DAGGER_TRIGGER_FRACS) and total_training_time >= TIME_BUDGET * DAGGER_TRIGGER_FRACS[dagger_round]:
        dagger_round += 1
        print(f"\n  DAgger round {dagger_round}/{len(DAGGER_TRIGGER_FRACS)}: collecting on-policy data at step {step}...")
        model.eval()
        new_examples = collect_dagger_data(model, tokenizer)
        model.train()
        all_dagger_examples.extend(new_examples)
        from prepare import load_dataset as _load_ds
        base_examples = _load_ds()["train_examples"]
        augmented = base_examples + all_dagger_examples
        train_loader = _make_dataloader_from_examples(tokenizer, augmented, DEVICE_BATCH_SIZE, TRAIN_SEQ_LEN)
        x, y, d, epoch = next(train_loader)
        print(f"  DAgger: +{len(new_examples)} examples (cumulative {len(all_dagger_examples)}, total {len(augmented)}). Resuming.")

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)
    metrics_writer.writerow({
        "step": step,
        "progress": progress,
        "loss": debiased_smooth_loss,
        "lr_multiplier": lrm,
        "dt_ms": dt * 1000,
        "tok_per_sec": tok_per_sec,
        "mfu_percent": mfu,
        "epoch": epoch,
        "remaining_seconds": remaining,
    })
    metrics_file.flush()
    loss_history.append((step, debiased_smooth_loss))

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Save model checkpoint
checkpoint_path = run_dir / "model.pt"
# Unwrap compiled model if needed
raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
torch.save({
    'model_state_dict': raw_model.state_dict(),
    'value_mlp_state_dict': value_mlp.state_dict(),
    'config': asdict(config),
    'step': step,
    'total_tokens': total_tokens,
}, checkpoint_path)
print(f"Saved checkpoint: {checkpoint_path}")

# Final eval
model.eval()
value_mlp.eval()
model.value_mlp = value_mlp  # attach for search use
with autocast_ctx:
    eval_metrics = evaluate_policy(model, tokenizer, DEVICE_BATCH_SIZE)

# Trajectory diagnostics: inspect a panel of held-out rollouts
from prepare import load_dataset, _cube_residual_error, _select_move_with_search
from rubiks import Cube, Episode

diag_payload = load_dataset()
diag_episodes = [Episode.from_dict(e) for e in diag_payload["eval_episodes"]["id"][:4]]
print("\n--- Trajectory diagnostics (4 held-out episodes) ---")
for ep_idx, episode in enumerate(diag_episodes):
    cube = Cube(episode.size)
    cube.apply_moves(episode.scramble)
    init_residual = _cube_residual_error(cube)
    moves_taken = []
    visited_states = {cube.to_kociemba_string()}
    outcome = "exhausted"
    for step_i in range(50):
        if cube.has_uniform_faces():
            outcome = "SOLVED"
            break
        move = _select_move_with_search(model, tokenizer, cube, moves_taken, visited_states)
        if move is None:
            outcome = "premature_done"
            break
        try:
            cube.apply_move(move)
        except Exception:
            outcome = "invalid_move"
            break
        moves_taken.append(move)
        visited_states.add(cube.to_kociemba_string())
    final_residual = _cube_residual_error(cube)
    print(f"  ep{ep_idx}: sol_len={len(episode.solution)} init_res={init_residual} final_res={final_residual} steps={len(moves_taken)} outcome={outcome}")
print()

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
if device_type == "cuda":
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    peak_vram_mb = 0.0

print("---")
print(f"primary_metric:   {eval_metrics['primary_metric']:.6f}")
print(f"id_move_acc:      {eval_metrics['id_move_accuracy']:.6f}")
print(f"ood_dev_move_acc: {eval_metrics['ood_dev_move_accuracy']:.6f}")
print(f"id_solve_rate:    {eval_metrics['id_solve_rate']:.6f}")
print(f"ood_dev_solve:    {eval_metrics['ood_dev_solve_rate']:.6f}")
print(f"ood_test_solve:   {eval_metrics['ood_test_solve_rate']:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
for split_name, metrics in eval_metrics["size_metrics"].items():
    if not metrics:
        continue
    size_summary = " ".join(
        f"{size}:{stats['solve_rate']:.2f}"
        for size, stats in sorted(metrics.items())
    )
    print(f"{split_name}_solve_by_size: {size_summary}")

metrics_file.close()

if loss_history:
    xs, ys = zip(*loss_history)
    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, color="#225c4a", linewidth=2)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Smoothed Loss")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=180)
    plt.close()

summary_payload = {
    **summary_base,
    "status": "completed",
    "finished_at": now_iso(),
    "results": {
        "primary_metric": eval_metrics["primary_metric"],
        "id_move_accuracy": eval_metrics["id_move_accuracy"],
        "ood_dev_move_accuracy": eval_metrics["ood_dev_move_accuracy"],
        "id_solve_rate": eval_metrics["id_solve_rate"],
        "ood_dev_solve_rate": eval_metrics["ood_dev_solve_rate"],
        "ood_test_solve_rate": eval_metrics["ood_test_solve_rate"],
        "training_seconds": total_training_time,
        "total_seconds": t_end - t_start,
        "peak_vram_mb": peak_vram_mb,
        "mfu_percent": steady_state_mfu,
        "total_tokens_m": total_tokens / 1e6,
        "num_steps": step,
        "num_params_m": num_params / 1e6,
    },
    "size_metrics": make_json_safe(eval_metrics["size_metrics"]),
}
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump(summary_payload, f, indent=2)

print(f"metrics_csv:      {metrics_csv_path}")
print(f"loss_plot:        {loss_plot_path}")
print(f"summary_json:     {summary_json_path}")
