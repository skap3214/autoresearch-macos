#!/usr/bin/env python3
"""
Interactive playground for the 2x2 Rubik's Cube neural solver.
Loads a trained checkpoint and serves a web UI with real-time solving.

Usage:
    python playground.py --checkpoint runs/<run>/model.pt
    python playground.py --checkpoint runs/<run>/model.pt --port 8080
"""

import argparse
import json
import os
import random
import sys
import webbrowser
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rubiks import Cube, Move, FACE_ORDER, random_scramble, scramble_length_for_size, build_prompt_tokens, build_answer_tokens, parse_answer_tokens

# ---------------------------------------------------------------------------
# Model Architecture (copied from train.py for standalone loading)
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
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
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
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin_ = cos_sin
        q, k = apply_rotary_emb(q, cos, sin_), apply_rotary_emb(k, cos, sin_)
        q, k = norm(q), norm(k)
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        window = window_size[0]
        if window > 0 and window < T:
            mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
            mask = mask.triu(diagonal=1 - window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


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
        self.value_head = nn.Linear(config.n_embd, 1, bias=True)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin_ = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin_, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin_ = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin_[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def _backbone(self, idx):
        B, T = idx.size()
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
        logits = self.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            policy_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                          ignore_index=-1, reduction=reduction)
            if distances is not None:
                sup_mask = (targets != -1).float()
                value_mask = torch.zeros_like(sup_mask)
                value_mask[:, :-1] = sup_mask[:, 1:]
                x_f32 = x.float()
                value_all = self.value_head(x_f32).squeeze(-1)
                value_pred = (value_all * value_mask).sum(dim=1)
                value_loss = F.mse_loss(value_pred, distances)
                return policy_loss + 0.5 * value_loss
            return policy_loss
        return logits

    def predict_value(self, idx):
        x = self._backbone(idx)
        return self.value_head(x[:, -1, :]).squeeze(-1)


class ValueMLP(nn.Module):
    def __init__(self, n_stickers=24, n_colors=6, hidden=256):
        super().__init__()
        self.embed = nn.Embedding(n_colors, 16)
        self.net = nn.Sequential(
            nn.Linear(n_stickers * 16, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, stickers):
        x = self.embed(stickers)
        x = x.view(x.size(0), -1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Inference helpers (from prepare.py, adapted for standalone use)
# ---------------------------------------------------------------------------

SEARCH_RESIDUAL_DELTA = 2
SEARCH_LOOKAHEAD_TOP_K = 3
ROLLOUT_MIN_STEPS = 200

from prepare import Tokenizer, TOKENIZER_DIR

def _autocast_ctx(device_type):
    if device_type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device_type == "cpu":
        return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def _cube_residual_error(cube):
    from collections import Counter
    error = 0
    for face in FACE_ORDER:
        colors = [c for row in cube.face_grid(face) for c in row]
        most_common_count = Counter(colors).most_common(1)[0][1]
        error += len(colors) - most_common_count
    return error


def _build_prompt_ids(tokenizer, cube, history):
    prompt_tokens = build_prompt_tokens(cube.size, cube, history=history)
    return [tokenizer.get_bos_token_id(), *tokenizer.encode_tokens(prompt_tokens)]


@torch.no_grad()
def select_move(model, tokenizer, cube, history, visited_states):
    """Select the next move using hybrid greedy search."""
    device = next(model.parameters()).device
    t2i = tokenizer.token_to_id
    ctx = _autocast_ctx(device.type)

    face_names = ("U", "R", "F", "D", "L", "B")
    turn_names = ("CW", "CCW", "HALF")
    turn_to_val = {"CW": 1, "CCW": -1, "HALF": 2}
    last_move = history[-1] if history else None

    prompt_ids = _build_prompt_ids(tokenizer, cube, history)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    with ctx:
        logits = model(input_ids)
    last_logits = logits[0, -1].float()

    current_residual = _cube_residual_error(cube)
    candidates = []

    for face in face_names:
        for turn_name in turn_names:
            turns = turn_to_val[turn_name]
            if last_move and face == last_move.face:
                inv = {1: -1, -1: 1, 2: 2}[last_move.turns]
                if turns == inv:
                    continue

            tid = t2i[f"MOVE_{face}_{turn_name}"]
            score = last_logits[tid].item()
            move = Move(face=face, depth=1, width=1, turns=turns)
            next_cube = cube.copy()
            next_cube.apply_move(move)
            next_state_str = next_cube.to_kociemba_string()
            if next_state_str in visited_states:
                continue

            candidates.append({
                "move": move,
                "score": score,
                "residual": _cube_residual_error(next_cube),
                "is_goal": next_cube.has_uniform_faces(),
            })

    if candidates:
        acceptable = [c for c in candidates if c["residual"] <= current_residual + SEARCH_RESIDUAL_DELTA]
        pool = acceptable if acceptable else candidates
        shortlist = sorted(pool, key=lambda c: c["score"], reverse=True)[:SEARCH_LOOKAHEAD_TOP_K]
        return shortlist[0]["move"]

    # Fallback: greedy without search
    done_id = t2i["<DONE>"]
    valid_ids = [done_id]
    for face in face_names:
        for turn in turn_names:
            valid_ids.append(t2i[f"MOVE_{face}_{turn}"])
    mask = torch.full_like(last_logits, float('-inf'))
    for vid in valid_ids:
        mask[vid] = 0.0
    chosen = int((last_logits + mask).argmax().item())
    token = tokenizer.id_to_token[chosen]
    try:
        return parse_answer_tokens([token])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(path, device="cpu"):
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint['config'])
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    if 'value_mlp_state_dict' in checkpoint:
        value_mlp = ValueMLP()
        value_mlp.load_state_dict(checkpoint['value_mlp_state_dict'])
        value_mlp.to(device)
        value_mlp.eval()
        model.value_mlp = value_mlp

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {config.n_layer} layers, {config.n_embd} dim, {n_params/1e6:.1f}M params")
    return model


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

class SolverState:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.rng = random.Random()
        self.reset()

    def reset(self):
        self.cube = Cube(2)
        self.history = []
        self.visited = set()
        self.visited.add(self.cube.to_kociemba_string())
        self.scramble_moves = []
        self.solve_moves = []
        self.solving = False

    def scramble(self, length=14):
        self.reset()
        self.scramble_moves = list(random_scramble(
            size=2, length=length, rng=self.rng,
            max_depth=2, max_width=2,
        ))
        self.cube.apply_moves(self.scramble_moves)
        self.visited = {self.cube.to_kociemba_string()}
        self.solving = False
        self.solve_moves = []

    def step(self):
        if self.cube.has_uniform_faces():
            return None, True

        move = select_move(
            self.model, self.tokenizer,
            self.cube, self.history, self.visited,
        )
        if move is None:
            return None, False

        self.cube.apply_move(move)
        self.history.append(move)
        self.visited.add(self.cube.to_kociemba_string())
        self.solve_moves.append(move)
        solved = self.cube.has_uniform_faces()
        return move, solved

    def get_face_grids(self):
        grids = {}
        for face in FACE_ORDER:
            grids[face] = self.cube.face_grid(face)
        return grids

    def to_json(self):
        return {
            "face_grids": self.get_face_grids(),
            "solved": self.cube.has_uniform_faces(),
            "step_count": len(self.solve_moves),
            "residual": _cube_residual_error(self.cube),
            "scramble_length": len(self.scramble_moves),
        }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>2x2 Rubik's Cube Neural Solver</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: #0f0f1a;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 30px 20px;
}
h1 {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, #4fc3f7, #7c4dff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.subtitle { color: #888; font-size: 14px; margin-bottom: 30px; }

.main {
    display: flex;
    gap: 40px;
    align-items: flex-start;
    flex-wrap: wrap;
    justify-content: center;
}

/* 3D Cube */
.scene {
    width: 220px;
    height: 220px;
    perspective: 600px;
    margin: 20px auto;
}
.cube-3d {
    width: 220px;
    height: 220px;
    position: relative;
    transform-style: preserve-3d;
    transform: rotateX(-25deg) rotateY(35deg);
    transition: transform 0.1s ease;
}
.face-3d {
    position: absolute;
    width: 220px;
    height: 220px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 6px;
    padding: 8px;
    background: #1a1a2e;
    border: 2px solid #2a2a4a;
    border-radius: 10px;
    backface-visibility: hidden;
}
.face-3d.front  { transform: translateZ(110px); }
.face-3d.back   { transform: rotateY(180deg) translateZ(110px); }
.face-3d.right  { transform: rotateY(90deg) translateZ(110px); }
.face-3d.left   { transform: rotateY(-90deg) translateZ(110px); }
.face-3d.top    { transform: rotateX(90deg) translateZ(110px); }
.face-3d.bottom { transform: rotateX(-90deg) translateZ(110px); }

.sticker-3d {
    border-radius: 8px;
    transition: background-color 0.25s ease;
    box-shadow: inset 0 0 0 1px rgba(0,0,0,0.3);
}

/* 2D Unfolded View */
.unfolded {
    display: grid;
    grid-template-columns: repeat(4, 56px);
    grid-template-rows: repeat(3, 56px);
    gap: 4px;
    margin: 20px auto;
}
.face-2d {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 3px;
    padding: 3px;
    background: #1a1a2e;
    border-radius: 6px;
}
.face-2d.u { grid-column: 2; grid-row: 1; }
.face-2d.l { grid-column: 1; grid-row: 2; }
.face-2d.f { grid-column: 2; grid-row: 2; }
.face-2d.r { grid-column: 3; grid-row: 2; }
.face-2d.b { grid-column: 4; grid-row: 2; }
.face-2d.d { grid-column: 2; grid-row: 3; }

.sticker-2d {
    width: 24px;
    height: 24px;
    border-radius: 4px;
    transition: background-color 0.25s ease;
    box-shadow: inset 0 0 0 1px rgba(0,0,0,0.2);
}

.face-label {
    position: absolute;
    font-size: 10px;
    color: rgba(255,255,255,0.5);
    font-weight: 700;
    pointer-events: none;
}

/* Panel */
.panel {
    background: #161625;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 24px;
    min-width: 280px;
}
.panel h2 { font-size: 16px; margin-bottom: 16px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }

.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #1e1e35;
}
.stat-label { color: #888; font-size: 14px; }
.stat-value { font-size: 14px; font-weight: 600; font-family: 'SF Mono', 'Cascadia Code', monospace; }

.controls { margin-top: 20px; display: flex; flex-direction: column; gap: 10px; }

.btn {
    padding: 10px 20px;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
}
.btn:hover { transform: translateY(-1px); }
.btn:active { transform: translateY(0); }
.btn-scramble {
    background: linear-gradient(135deg, #7c4dff, #536dfe);
    color: white;
}
.btn-scramble:hover { box-shadow: 0 4px 15px rgba(124, 77, 255, 0.4); }
.btn-solve {
    background: linear-gradient(135deg, #00c853, #00e676);
    color: #0a0a0a;
}
.btn-solve:hover { box-shadow: 0 4px 15px rgba(0, 200, 83, 0.4); }
.btn-solve:disabled {
    background: #333;
    color: #666;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
}
.btn-reset {
    background: #2a2a4a;
    color: #aaa;
}

.speed-control {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 4px;
}
.speed-control label { font-size: 13px; color: #888; }
.speed-control input[type=range] { flex: 1; accent-color: #7c4dff; }
.speed-control .speed-val { font-size: 12px; color: #aaa; font-family: monospace; min-width: 40px; }

.scramble-control {
    display: flex;
    align-items: center;
    gap: 10px;
}
.scramble-control label { font-size: 13px; color: #888; }
.scramble-control input[type=range] { flex: 1; accent-color: #7c4dff; }
.scramble-control .scramble-val { font-size: 12px; color: #aaa; font-family: monospace; min-width: 20px; }

/* Move history */
.move-history {
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid #1e1e35;
}
.move-history h3 { font-size: 13px; color: #666; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }
.moves-list {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    max-height: 120px;
    overflow-y: auto;
    font-family: 'SF Mono', 'Cascadia Code', monospace;
    font-size: 12px;
}
.move-tag {
    padding: 3px 8px;
    background: #1e1e35;
    border-radius: 4px;
    color: #7c4dff;
    white-space: nowrap;
}
.move-tag.latest {
    background: #2a1e55;
    color: #b388ff;
    font-weight: 700;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.status-badge.solved { background: #1b5e20; color: #69f0ae; }
.status-badge.scrambled { background: #4a1c00; color: #ffab40; }
.status-badge.solving { background: #1a237e; color: #82b1ff; animation: pulse 1s infinite; }
.status-badge.ready { background: #1e1e35; color: #888; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* Drag rotation */
.scene { cursor: grab; }
.scene:active { cursor: grabbing; }
</style>
</head>
<body>
<h1>2x2 Rubik's Cube Neural Solver</h1>
<p class="subtitle">25.4M parameter transformer trained on 615K examples via imitation learning + DAgger</p>

<div class="main">
    <div>
        <div class="scene" id="scene">
            <div class="cube-3d" id="cube3d">
                <div class="face-3d front" id="face-F"></div>
                <div class="face-3d back" id="face-B"></div>
                <div class="face-3d right" id="face-R"></div>
                <div class="face-3d left" id="face-L"></div>
                <div class="face-3d top" id="face-U"></div>
                <div class="face-3d bottom" id="face-D"></div>
            </div>
        </div>
        <div class="unfolded" id="unfolded">
            <div class="face-2d u" id="flat-U"></div>
            <div class="face-2d l" id="flat-L"></div>
            <div class="face-2d f" id="flat-F"></div>
            <div class="face-2d r" id="flat-R"></div>
            <div class="face-2d b" id="flat-B"></div>
            <div class="face-2d d" id="flat-D"></div>
        </div>
    </div>

    <div class="panel">
        <h2>Controls</h2>

        <div class="stat-row">
            <span class="stat-label">Status</span>
            <span id="status" class="status-badge ready">Ready</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Steps</span>
            <span class="stat-value" id="step-count">0</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Residual</span>
            <span class="stat-value" id="residual">0</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Scramble Length</span>
            <span class="stat-value" id="scramble-len">-</span>
        </div>

        <div class="controls">
            <div class="scramble-control">
                <label>Moves:</label>
                <input type="range" id="scramble-depth" min="4" max="30" value="14">
                <span class="scramble-val" id="scramble-depth-val">14</span>
            </div>
            <button class="btn btn-scramble" id="btn-scramble" onclick="doScramble()">Scramble</button>
            <button class="btn btn-solve" id="btn-solve" onclick="doSolve()" disabled>Solve</button>
            <button class="btn btn-reset" onclick="doReset()">Reset</button>

            <div class="speed-control">
                <label>Speed:</label>
                <input type="range" id="speed" min="50" max="1000" value="300" step="50">
                <span class="speed-val" id="speed-val">300ms</span>
            </div>
        </div>

        <div class="move-history">
            <h3>Solve Moves</h3>
            <div class="moves-list" id="moves-list"></div>
        </div>
    </div>
</div>

<script>
const COLORS = {
    'W': '#ffffff', 'Y': '#ffd500', 'G': '#009b48',
    'B': '#0046ad', 'R': '#b71234', 'O': '#ff5800'
};

// Initialize stickers
const faces3d = ['U', 'R', 'F', 'D', 'L', 'B'];
const faceMap3d = { 'F': 'face-F', 'B': 'face-B', 'R': 'face-R', 'L': 'face-L', 'U': 'face-U', 'D': 'face-D' };
const faceMap2d = { 'U': 'flat-U', 'R': 'flat-R', 'F': 'flat-F', 'D': 'flat-D', 'L': 'flat-L', 'B': 'flat-B' };

function initStickers() {
    for (const face of faces3d) {
        const el3d = document.getElementById(faceMap3d[face]);
        const el2d = document.getElementById(faceMap2d[face]);
        el3d.innerHTML = '';
        el2d.innerHTML = '';
        for (let i = 0; i < 4; i++) {
            const s3d = document.createElement('div');
            s3d.className = 'sticker-3d';
            s3d.id = `s3d-${face}-${i}`;
            el3d.appendChild(s3d);

            const s2d = document.createElement('div');
            s2d.className = 'sticker-2d';
            s2d.id = `s2d-${face}-${i}`;
            el2d.appendChild(s2d);
        }
    }
}

function updateCube(faceGrids) {
    for (const face of faces3d) {
        const grid = faceGrids[face];
        for (let r = 0; r < 2; r++) {
            for (let c = 0; c < 2; c++) {
                const idx = r * 2 + c;
                const color = COLORS[grid[r][c]];
                document.getElementById(`s3d-${face}-${idx}`).style.backgroundColor = color;
                document.getElementById(`s2d-${face}-${idx}`).style.backgroundColor = color;
            }
        }
    }
}

function updateStats(data) {
    document.getElementById('step-count').textContent = data.step_count;
    document.getElementById('residual').textContent = data.residual;
    document.getElementById('scramble-len').textContent = data.scramble_length || '-';

    const badge = document.getElementById('status');
    if (data.solved) {
        badge.className = 'status-badge solved';
        badge.textContent = 'Solved!';
    } else if (solving) {
        badge.className = 'status-badge solving';
        badge.textContent = 'Solving...';
    } else if (data.step_count === 0 && data.scramble_length > 0) {
        badge.className = 'status-badge scrambled';
        badge.textContent = 'Scrambled';
    } else {
        badge.className = 'status-badge ready';
        badge.textContent = 'Ready';
    }
}

let solving = false;
let solveAbort = false;

async function api(endpoint, body = {}) {
    const resp = await fetch(`/api/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    return resp.json();
}

async function doScramble() {
    solveAbort = true;
    solving = false;
    const length = parseInt(document.getElementById('scramble-depth').value);
    const data = await api('scramble', { length });
    updateCube(data.face_grids);
    updateStats(data);
    document.getElementById('moves-list').innerHTML = '';
    document.getElementById('btn-solve').disabled = false;
}

async function doSolve() {
    if (solving) return;
    solving = true;
    solveAbort = false;
    document.getElementById('btn-solve').disabled = true;

    const badge = document.getElementById('status');
    badge.className = 'status-badge solving';
    badge.textContent = 'Solving...';

    const movesList = document.getElementById('moves-list');
    const speed = parseInt(document.getElementById('speed').value);

    for (let i = 0; i < ROLLOUT_MAX; i++) {
        if (solveAbort) break;
        const data = await api('step');

        updateCube(data.face_grids);
        updateStats(data);

        if (data.move) {
            // Add move tag
            // Remove latest class from previous
            const prev = movesList.querySelector('.latest');
            if (prev) prev.classList.remove('latest');
            const tag = document.createElement('span');
            tag.className = 'move-tag latest';
            tag.textContent = data.move;
            movesList.appendChild(tag);
            movesList.scrollTop = movesList.scrollHeight;
        }

        if (data.solved) {
            badge.className = 'status-badge solved';
            badge.textContent = 'Solved!';
            solving = false;
            return;
        }

        if (!data.move) {
            solving = false;
            return;
        }

        await new Promise(r => setTimeout(r, speed));
    }
    solving = false;
}

async function doReset() {
    solveAbort = true;
    solving = false;
    const data = await api('reset');
    updateCube(data.face_grids);
    updateStats(data);
    document.getElementById('moves-list').innerHTML = '';
    document.getElementById('btn-solve').disabled = true;
}

const ROLLOUT_MAX = 200;

// Speed slider
document.getElementById('speed').addEventListener('input', (e) => {
    document.getElementById('speed-val').textContent = e.target.value + 'ms';
});
document.getElementById('scramble-depth').addEventListener('input', (e) => {
    document.getElementById('scramble-depth-val').textContent = e.target.value;
});

// Drag to rotate 3D cube
let isDragging = false;
let prevX = 0, prevY = 0;
let rotX = -25, rotY = 35;
const scene = document.getElementById('scene');
const cube3d = document.getElementById('cube3d');

scene.addEventListener('mousedown', (e) => {
    isDragging = true;
    prevX = e.clientX;
    prevY = e.clientY;
});
window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const dx = e.clientX - prevX;
    const dy = e.clientY - prevY;
    rotY += dx * 0.5;
    rotX -= dy * 0.5;
    cube3d.style.transform = `rotateX(${rotX}deg) rotateY(${rotY}deg)`;
    prevX = e.clientX;
    prevY = e.clientY;
});
window.addEventListener('mouseup', () => { isDragging = false; });

// Init
initStickers();
(async () => {
    const data = await api('reset');
    updateCube(data.face_grids);
    updateStats(data);
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------

state = None  # initialized in main()

class PlaygroundHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        else:
            self.send_error(404)

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}

        if self.path == '/api/scramble':
            length = body.get('length', 14)
            state.scramble(length)
            result = state.to_json()

        elif self.path == '/api/step':
            move, solved = state.step()
            result = state.to_json()
            if move:
                result['move'] = f"{move.face} {move.turn_name()}"
            else:
                result['move'] = None

        elif self.path == '/api/reset':
            state.reset()
            result = state.to_json()

        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def log_message(self, format, *args):
        pass  # suppress request logs


def find_latest_checkpoint():
    runs_dir = Path(__file__).parent / "runs"
    if not runs_dir.exists():
        return None
    checkpoints = sorted(runs_dir.glob("*/model.pt"), key=lambda p: p.parent.name, reverse=True)
    return str(checkpoints[0]) if checkpoints else None


def main():
    parser = argparse.ArgumentParser(description="2x2 Rubik's Cube Solver Playground")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model.pt checkpoint')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu/auto)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser')
    args = parser.parse_args()

    # Find checkpoint
    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    if checkpoint_path is None:
        print("Error: No checkpoint found. Run training first, or specify --checkpoint path.")
        sys.exit(1)

    # Select device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load model
    model = load_checkpoint(checkpoint_path, device=device)
    tokenizer = Tokenizer.from_directory()

    global state
    state = SolverState(model, tokenizer)

    # Start server
    server = HTTPServer(('0.0.0.0', args.port), PlaygroundHandler)
    url = f"http://localhost:{args.port}"
    print(f"\nPlayground running at {url}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()


if __name__ == '__main__':
    main()
