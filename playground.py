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
import threading
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
        self.cube_size = 2
        self.reset()

    def reset(self, size=None):
        if size is not None:
            self.cube_size = size
        self.cube = Cube(self.cube_size)
        self.history = []
        self.visited = set()
        self.visited.add(self.cube.to_kociemba_string())
        self.scramble_moves = []
        self.solve_moves = []

    def is_goal(self):
        if self.cube_size == 2:
            return self.cube.has_uniform_faces()
        return self.cube.is_solved()

    def scramble(self, length=14, size=None):
        self.reset(size=size)
        max_d = min(2, self.cube_size // 2) if self.cube_size <= 3 else 2
        max_w = min(2, self.cube_size // 2) if self.cube_size <= 3 else 2
        self.scramble_moves = list(random_scramble(
            size=self.cube_size, length=length, rng=self.rng,
            max_depth=max_d, max_width=max_w,
        ))
        self.cube.apply_moves(self.scramble_moves)
        self.visited = {self.cube.to_kociemba_string()}
        self.solve_moves = []

    def step(self):
        if self.is_goal():
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
        return move, self.is_goal()

    def solve_all(self):
        """Solve greedily, returning all (move, face_grids) snapshots for replay."""
        snapshots = []
        for _ in range(ROLLOUT_MIN_STEPS):
            if self.is_goal():
                break
            move = select_move(
                self.model, self.tokenizer,
                self.cube, self.history, self.visited,
            )
            if move is None:
                break
            self.cube.apply_move(move)
            self.history.append(move)
            self.visited.add(self.cube.to_kociemba_string())
            self.solve_moves.append(move)
            snapshots.append({
                "move": f"{move.face} {move.turn_name()}",
                "face_grids": self.get_face_grids(),
                "solved": self.is_goal(),
                "step_count": len(self.solve_moves),
                "residual": _cube_residual_error(self.cube),
            })
            if self.is_goal():
                break
        return snapshots

    def start_beam_search(self, beam_width=32, max_steps=100):
        """Start beam search in a background thread, updating self.beam_progress."""
        self.beam_progress = {
            "status": "running",
            "step": 0,
            "max_steps": max_steps,
            "active_beams": 0,
            "best_value": 0.0,
            "states_explored": 0,
            "best_face_grids": self.get_face_grids(),
            "best_residual": _cube_residual_error(self.cube),
            "snapshots": None,
        }

        def _run():
            try:
                snapshots = self._beam_search_inner(beam_width, max_steps)
                self.beam_progress["status"] = "done"
                self.beam_progress["snapshots"] = snapshots
            except Exception as e:
                self.beam_progress["status"] = "error"
                self.beam_progress["snapshots"] = []

        self._beam_thread = threading.Thread(target=_run, daemon=True)
        self._beam_thread.start()

    def _beam_search_inner(self, beam_width, max_steps):
        from prepare import _enumerate_move_candidates, _build_prompt_ids, _autocast_context
        import torch

        device = next(self.model.parameters()).device
        autocast_ctx = _autocast_context(device.type)

        if self.is_goal():
            return []

        initial_state = self.cube.to_kociemba_string()
        beams = [(self.cube.copy(), [], {initial_state}, 0.0, [])]
        total_states = 0

        for step in range(max_steps):
            if not beams:
                break

            all_candidates = []
            for beam_idx, (b_cube, b_history, b_visited, b_score, b_trace) in enumerate(beams):
                if b_cube.has_uniform_faces() if b_cube.size == 2 else b_cube.is_solved():
                    return self._trace_to_snapshots(b_trace)

                candidates = _enumerate_move_candidates(
                    self.model, self.tokenizer, b_cube, b_history, b_visited)
                for c in candidates:
                    if c["is_goal"]:
                        trace = b_trace + [c["move"]]
                        return self._trace_to_snapshots(trace)
                    all_candidates.append((beam_idx, c))

            if not all_candidates:
                break

            total_states += len(all_candidates)

            # Batch evaluate with value head
            prompt_ids_list = []
            for beam_idx, c in all_candidates:
                b_history = beams[beam_idx][1]
                next_history = [*b_history, c["move"]]
                ids = _build_prompt_ids(self.tokenizer, c["next_cube"], next_history)
                prompt_ids_list.append(ids)

            max_len = max(len(ids) for ids in prompt_ids_list)
            pad_id = self.tokenizer.get_pad_token_id()
            batch = torch.full((len(prompt_ids_list), max_len), pad_id,
                               dtype=torch.long, device=device)
            for i, ids in enumerate(prompt_ids_list):
                batch[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

            with autocast_ctx:
                values = self.model.predict_value(batch).float()

            scored = []
            for i, (beam_idx, c) in enumerate(all_candidates):
                parent_score = beams[beam_idx][3]
                candidate_score = parent_score + c["score"] - 0.5 * values[i].item()
                scored.append((candidate_score, beam_idx, c, values[i].item()))

            scored.sort(key=lambda x: -x[0])
            new_beams = []
            seen_states = set()
            for score, beam_idx, c, val in scored:
                if len(new_beams) >= beam_width:
                    break
                state_str = c["next_state_str"]
                if state_str in seen_states:
                    continue
                seen_states.add(state_str)

                parent = beams[beam_idx]
                new_history = [*parent[1], c["move"]]
                new_visited = set(parent[2])
                new_visited.add(state_str)
                new_trace = parent[4] + [c["move"]]
                new_beams.append((c["next_cube"], new_history, new_visited, score, new_trace))

            beams = new_beams

            # Update progress with best beam's state
            if beams:
                best_cube = beams[0][0]
                best_grids = {}
                for face in FACE_ORDER:
                    best_grids[face] = best_cube.face_grid(face)
                best_val = scored[0][3] if scored else 0
                self.beam_progress.update({
                    "step": step + 1,
                    "active_beams": len(beams),
                    "best_value": round(best_val, 1),
                    "states_explored": total_states,
                    "best_face_grids": best_grids,
                    "best_residual": _cube_residual_error(best_cube),
                })

        # Check final beams
        for b_cube, _, _, _, b_trace in beams:
            if b_cube.has_uniform_faces() if b_cube.size == 2 else b_cube.is_solved():
                return self._trace_to_snapshots(b_trace)

        if beams:
            return self._trace_to_snapshots(beams[0][4])
        return []

    def _trace_to_snapshots(self, moves):
        """Replay a move trace on the cube and return snapshots."""
        # Reset cube to scrambled state
        self.cube = Cube(self.cube_size)
        self.cube.apply_moves(self.scramble_moves)
        self.history = []
        self.visited = {self.cube.to_kociemba_string()}
        self.solve_moves = []

        snapshots = []
        for move in moves:
            self.cube.apply_move(move)
            self.history.append(move)
            self.visited.add(self.cube.to_kociemba_string())
            self.solve_moves.append(move)
            snapshots.append({
                "move": f"{move.face} {move.turn_name()}",
                "face_grids": self.get_face_grids(),
                "solved": self.is_goal(),
                "step_count": len(self.solve_moves),
                "residual": _cube_residual_error(self.cube),
            })
            if self.is_goal():
                break
        return snapshots

    def get_face_grids(self):
        grids = {}
        for face in FACE_ORDER:
            grids[face] = self.cube.face_grid(face)
        return grids

    def to_json(self):
        return {
            "face_grids": self.get_face_grids(),
            "solved": self.is_goal(),
            "step_count": len(self.solve_moves),
            "residual": _cube_residual_error(self.cube),
            "scramble_length": len(self.scramble_moves),
            "cube_size": self.cube_size,
        }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neural Cube Solver</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#060608;--surface:#0e0e14;--border:#1a1a28;
  --text:#c8c5d0;--text-dim:#5a5872;--text-bright:#f0eef5;
  --amber:#e8a830;--amber-dim:#7a5a1a;--amber-glow:rgba(232,168,48,0.12);
  --green:#3dd68c;--green-dim:#1a4a32;
  --red:#e85050;
  --blue:#5088e8;
  --radius:10px;
}
html{font-size:15px}
body{
  font-family:'Outfit',sans-serif;font-weight:400;
  background:var(--bg);color:var(--text);
  min-height:100vh;overflow-x:hidden;
  background-image:
    radial-gradient(ellipse 80% 60% at 50% 0%,rgba(232,168,48,0.04),transparent),
    radial-gradient(ellipse 60% 40% at 80% 100%,rgba(80,136,232,0.03),transparent);
}
.container{
  max-width:960px;margin:0 auto;padding:48px 24px 64px;
  display:flex;flex-direction:column;align-items:center;
}

/* Header */
header{text-align:center;margin-bottom:48px}
header h1{
  font-family:'DM Serif Display',serif;font-size:2.6rem;font-weight:400;
  color:var(--text-bright);letter-spacing:-0.02em;line-height:1.1;
  margin-bottom:8px;
}
header h1 span{color:var(--amber)}
header p{color:var(--text-dim);font-size:0.82rem;letter-spacing:0.04em;font-weight:300}

/* Layout */
.layout{
  display:grid;grid-template-columns:1fr 300px;gap:48px;
  width:100%;align-items:start;
}
@media(max-width:760px){.layout{grid-template-columns:1fr;justify-items:center}}

.cube-area{display:flex;flex-direction:column;align-items:center;gap:28px}

/* 3D Cube */
.scene{
  width:280px;height:280px;perspective:700px;cursor:grab;
  filter:drop-shadow(0 20px 60px rgba(232,168,48,0.08));
}
.scene:active{cursor:grabbing}
.cube-3d{
  width:280px;height:280px;position:relative;
  transform-style:preserve-3d;
  transform:rotateX(-25deg) rotateY(35deg);
  transition:transform 0.08s linear;
}
.face-3d{
  position:absolute;width:280px;height:280px;
  display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;
  gap:8px;padding:10px;
  background:rgba(14,14,20,0.92);
  border:1.5px solid var(--border);border-radius:14px;
  backface-visibility:hidden;
}
.face-3d.front{transform:translateZ(140px)}
.face-3d.back{transform:rotateY(180deg) translateZ(140px)}
.face-3d.right{transform:rotateY(90deg) translateZ(140px)}
.face-3d.left{transform:rotateY(-90deg) translateZ(140px)}
.face-3d.top{transform:rotateX(90deg) translateZ(140px)}
.face-3d.bottom{transform:rotateX(-90deg) translateZ(140px)}

.sticker-3d{
  border-radius:10px;
  transition:background-color 0.2s ease,box-shadow 0.3s ease;
  box-shadow:inset 0 1px 2px rgba(255,255,255,0.15),0 2px 8px rgba(0,0,0,0.4);
}

/* 2D Flat net */
.unfolded{
  display:grid;grid-template-columns:repeat(4,48px);grid-template-rows:repeat(3,48px);
  gap:3px;opacity:0.7;transition:opacity 0.3s;
}
.unfolded:hover{opacity:1}
.face-2d{
  display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;
  gap:2px;padding:2px;background:var(--surface);border-radius:5px;
}
.face-2d.u{grid-column:2;grid-row:1}
.face-2d.l{grid-column:1;grid-row:2}
.face-2d.f{grid-column:2;grid-row:2}
.face-2d.r{grid-column:3;grid-row:2}
.face-2d.b{grid-column:4;grid-row:2}
.face-2d.d{grid-column:2;grid-row:3}
.sticker-2d{
  width:20px;height:20px;border-radius:3px;
  transition:background-color 0.2s ease;
}

/* Panel */
.panel{
  background:var(--surface);border:1px solid var(--border);
  border-radius:16px;padding:28px;
  display:flex;flex-direction:column;gap:20px;
}
.panel-title{
  font-family:'DM Serif Display',serif;font-size:1.15rem;
  color:var(--text-bright);margin-bottom:4px;
}

.stats{display:flex;flex-direction:column;gap:2px}
.stat-row{
  display:flex;justify-content:space-between;align-items:center;
  padding:9px 0;border-bottom:1px solid var(--border);
}
.stat-row:last-child{border-bottom:none}
.stat-label{color:var(--text-dim);font-size:0.82rem;font-weight:300}
.stat-value{
  font-family:'JetBrains Mono',monospace;font-size:0.82rem;
  font-weight:600;color:var(--text-bright);
}

/* Buttons */
.controls{display:flex;flex-direction:column;gap:8px}
.btn{
  padding:11px 20px;border:none;border-radius:var(--radius);
  font-family:'Outfit',sans-serif;font-size:0.88rem;font-weight:500;
  cursor:pointer;transition:all 0.2s ease;letter-spacing:0.02em;
  position:relative;overflow:hidden;
}
.btn::after{
  content:'';position:absolute;inset:0;
  background:linear-gradient(180deg,rgba(255,255,255,0.06),transparent);
  pointer-events:none;
}
.btn:hover{transform:translateY(-1px)}
.btn:active{transform:translateY(0)}

.btn-scramble{
  background:var(--amber);color:var(--bg);font-weight:600;
}
.btn-scramble:hover{box-shadow:0 6px 24px rgba(232,168,48,0.3)}

.btn-solve{background:var(--green);color:var(--bg);font-weight:600}
.btn-solve:hover{box-shadow:0 6px 24px rgba(61,214,140,0.3)}
.btn-solve:disabled{
  background:var(--border);color:var(--text-dim);
  cursor:not-allowed;transform:none;box-shadow:none;
}

.btn-reset{background:transparent;color:var(--text-dim);border:1px solid var(--border)}
.btn-reset:hover{border-color:var(--text-dim);color:var(--text)}

/* Sliders */
.slider-row{
  display:flex;align-items:center;gap:10px;
}
.slider-row label{font-size:0.78rem;color:var(--text-dim);min-width:44px;font-weight:300}
.slider-row input[type=range]{
  flex:1;height:4px;-webkit-appearance:none;appearance:none;
  background:var(--border);border-radius:2px;outline:none;
}
.slider-row input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:14px;height:14px;
  background:var(--amber);border-radius:50%;cursor:pointer;
  box-shadow:0 0 8px rgba(232,168,48,0.4);
}
.slider-val{
  font-family:'JetBrains Mono',monospace;font-size:0.72rem;
  color:var(--text-dim);min-width:36px;text-align:right;
}

/* Move history */
.move-history h3{
  font-size:0.72rem;color:var(--text-dim);
  text-transform:uppercase;letter-spacing:0.1em;font-weight:400;
  margin-bottom:8px;
}
.moves-list{
  display:flex;flex-wrap:wrap;gap:4px;
  max-height:100px;overflow-y:auto;
  font-family:'JetBrains Mono',monospace;font-size:0.72rem;
}
.moves-list::-webkit-scrollbar{width:3px}
.moves-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.move-tag{
  padding:3px 7px;background:var(--border);border-radius:4px;
  color:var(--amber);white-space:nowrap;
  transition:all 0.15s ease;
}
.move-tag.latest{
  background:var(--amber-dim);color:var(--amber);
  font-weight:600;box-shadow:0 0 10px rgba(232,168,48,0.15);
}

/* Status badge */
.status-badge{
  display:inline-block;padding:4px 12px;border-radius:20px;
  font-family:'JetBrains Mono',monospace;
  font-size:0.68rem;font-weight:600;
  text-transform:uppercase;letter-spacing:0.08em;
}
.status-badge.solved{background:var(--green-dim);color:var(--green)}
.status-badge.scrambled{background:rgba(232,168,48,0.12);color:var(--amber)}
.status-badge.solving{background:rgba(80,136,232,0.12);color:var(--blue);animation:pulse 1.2s ease infinite}
.status-badge.ready{background:var(--border);color:var(--text-dim)}

@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}

/* Solved celebration */
@keyframes celebrate{
  0%{filter:drop-shadow(0 20px 60px rgba(232,168,48,0.08))}
  50%{filter:drop-shadow(0 20px 80px rgba(61,214,140,0.25))}
  100%{filter:drop-shadow(0 20px 60px rgba(232,168,48,0.08))}
}
.scene.solved-glow{animation:celebrate 1.5s ease 3}

.face-label{position:absolute;font-size:10px;color:rgba(255,255,255,0.5);font-weight:700;pointer-events:none}

/* Size toggle */
.size-toggle{display:flex;gap:4px;margin-bottom:4px}
.size-btn{
  flex:1;padding:9px;border:1px solid var(--border);border-radius:var(--radius);
  background:transparent;color:var(--text-dim);font-family:'Outfit',sans-serif;
  font-size:0.85rem;font-weight:500;cursor:pointer;transition:all 0.2s;
}
.size-btn.active{background:var(--amber-dim);color:var(--amber);border-color:var(--amber)}
.size-btn:hover:not(.active){border-color:var(--text-dim)}

/* Timer */
.solve-timer{
  font-family:'JetBrains Mono',monospace;font-size:0.72rem;
  color:var(--amber);margin-top:4px;min-height:1em;
}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>Neural <span>Cube</span> Solver</h1>
  <p>85.4M parameter transformer &middot; 2x2 &amp; 3x3 &middot; trained via imitation learning + DAgger</p>
</header>

<div class="layout">
  <div class="cube-area">
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
    <div class="panel-title">Controls</div>

    <div class="stats">
      <div class="stat-row">
        <span class="stat-label">Status</span>
        <span id="status" class="status-badge ready">Ready</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Steps taken</span>
        <span class="stat-value" id="step-count">0</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Residual error</span>
        <span class="stat-value" id="residual">0</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Scramble depth</span>
        <span class="stat-value" id="scramble-len">&mdash;</span>
      </div>
    </div>

    <div class="controls">
      <div class="size-toggle">
        <button class="size-btn active" id="size-2" onclick="setSize(2)">2x2</button>
        <button class="size-btn" id="size-3" onclick="setSize(3)">3x3</button>
      </div>
      <div class="slider-row">
        <label>Moves</label>
        <input type="range" id="scramble-depth" min="4" max="30" value="14">
        <span class="slider-val" id="scramble-depth-val">14</span>
      </div>
      <button class="btn btn-scramble" id="btn-scramble" onclick="doScramble()">Scramble</button>
      <button class="btn btn-solve" id="btn-solve" onclick="doSolveRealtime()" disabled>Solve (fast)</button>
      <button class="btn btn-solve" id="btn-beam" onclick="doSolveBeam()" disabled style="background:var(--blue)">Solve (thorough)</button>
      <button class="btn btn-reset" onclick="doReset()">Reset</button>
      <div class="slider-row">
        <label>Replay</label>
        <input type="range" id="speed" min="0" max="500" value="80" step="10">
        <span class="slider-val" id="speed-val">80ms</span>
      </div>
    </div>

    <div class="solve-timer" id="solve-timer"></div>

    <div class="move-history">
      <h3>Solution trace</h3>
      <div class="moves-list" id="moves-list"></div>
    </div>
  </div>
</div>

</div>

<script>
const COLORS={
  W:'#f0eff4',Y:'#f5cc00',G:'#1da34d',
  B:'#2463c4',R:'#cc2936',O:'#e87020'
};
const faces3d=['U','R','F','D','L','B'];
const faceMap3d={F:'face-F',B:'face-B',R:'face-R',L:'face-L',U:'face-U',D:'face-D'};
const faceMap2d={U:'flat-U',R:'flat-R',F:'flat-F',D:'flat-D',L:'flat-L',B:'flat-B'};

let cubeSize=2, solving=false, solveAbort=false;

function initStickers(n){
  cubeSize=n;
  // 3D faces
  for(const face of faces3d){
    const el=document.getElementById(faceMap3d[face]);
    el.innerHTML='';
    el.style.gridTemplateColumns=`repeat(${n},1fr)`;
    el.style.gridTemplateRows=`repeat(${n},1fr)`;
    for(let i=0;i<n*n;i++){
      const s=document.createElement('div');
      s.className='sticker-3d';s.id=`s3d-${face}-${i}`;el.appendChild(s);
    }
  }
  // 2D flat
  const unfoldedEl=document.getElementById('unfolded');
  const stickerSize=n===2?48:32;
  unfoldedEl.style.gridTemplateColumns=`repeat(4,${stickerSize}px)`;
  unfoldedEl.style.gridTemplateRows=`repeat(3,${stickerSize}px)`;
  for(const face of faces3d){
    const el=document.getElementById(faceMap2d[face]);
    el.innerHTML='';
    el.style.gridTemplateColumns=`repeat(${n},1fr)`;
    el.style.gridTemplateRows=`repeat(${n},1fr)`;
    for(let i=0;i<n*n;i++){
      const s=document.createElement('div');
      s.className='sticker-2d';s.id=`s2d-${face}-${i}`;
      s.style.width=`${Math.floor((stickerSize-n*2-4)/n)}px`;
      s.style.height=s.style.width;
      el.appendChild(s);
    }
  }
  // Adjust 3D cube size
  const cubeW=n===2?280:280;
  const half=cubeW/2;
  document.querySelectorAll('.face-3d').forEach(f=>{f.style.width=f.style.height=cubeW+'px'});
  document.querySelector('.face-3d.front').style.transform=`translateZ(${half}px)`;
  document.querySelector('.face-3d.back').style.transform=`rotateY(180deg) translateZ(${half}px)`;
  document.querySelector('.face-3d.right').style.transform=`rotateY(90deg) translateZ(${half}px)`;
  document.querySelector('.face-3d.left').style.transform=`rotateY(-90deg) translateZ(${half}px)`;
  document.querySelector('.face-3d.top').style.transform=`rotateX(90deg) translateZ(${half}px)`;
  document.querySelector('.face-3d.bottom').style.transform=`rotateX(-90deg) translateZ(${half}px)`;
}

function updateCube(fg){
  for(const face of faces3d){
    const grid=fg[face]; const n=grid.length;
    for(let r=0;r<n;r++)for(let c=0;c<n;c++){
      const idx=r*n+c,color=COLORS[grid[r][c]];
      const s3=document.getElementById(`s3d-${face}-${idx}`);
      const s2=document.getElementById(`s2d-${face}-${idx}`);
      if(s3)s3.style.backgroundColor=color;
      if(s2)s2.style.backgroundColor=color;
    }
  }
}

function updateStats(data){
  document.getElementById('step-count').textContent=data.step_count;
  document.getElementById('residual').textContent=data.residual;
  document.getElementById('scramble-len').textContent=data.scramble_length||'\u2014';
  const badge=document.getElementById('status');
  const sc=document.getElementById('scene');
  if(data.solved){
    badge.className='status-badge solved';badge.textContent='Solved';sc.classList.add('solved-glow');
  }else if(solving){
    badge.className='status-badge solving';badge.textContent='Solving\u2026';sc.classList.remove('solved-glow');
  }else if(data.step_count===0&&data.scramble_length>0){
    badge.className='status-badge scrambled';badge.textContent='Scrambled';sc.classList.remove('solved-glow');
  }else{
    badge.className='status-badge ready';badge.textContent='Ready';sc.classList.remove('solved-glow');
  }
}

async function api(endpoint,body={}){
  const r=await fetch(`/api/${endpoint}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  return r.json();
}

async function setSize(n){
  solveAbort=true;solving=false;
  document.querySelectorAll('.size-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById(`size-${n}`).classList.add('active');
  initStickers(n);
  const data=await api('reset',{size:n});
  updateCube(data.face_grids);updateStats(data);
  document.getElementById('moves-list').innerHTML='';
  document.getElementById('solve-timer').textContent='';
  document.getElementById('btn-solve').disabled=true;
  document.getElementById('btn-beam').disabled=true;
}

async function doScramble(){
  solveAbort=true;solving=false;
  const length=parseInt(document.getElementById('scramble-depth').value);
  const data=await api('scramble',{length,size:cubeSize});
  initStickers(cubeSize);
  updateCube(data.face_grids);updateStats(data);
  document.getElementById('moves-list').innerHTML='';
  document.getElementById('solve-timer').textContent='';
  document.getElementById('btn-solve').disabled=false;
  document.getElementById('btn-beam').disabled=false;
}

async function doSolveRealtime(){
  if(solving)return;solving=true;solveAbort=false;
  document.getElementById('btn-solve').disabled=true;
  const badge=document.getElementById('status');
  badge.className='status-badge solving';badge.textContent='Thinking\u2026';
  const timer=document.getElementById('solve-timer');
  const movesList=document.getElementById('moves-list');
  const t0=performance.now();
  timer.textContent='Computing solution\u2026';

  // Server computes entire solution at once (real-time, no artificial delay)
  const result=await api('solve_all');
  const elapsed=((performance.now()-t0)/1000).toFixed(2);
  const snapshots=result.snapshots;

  if(!snapshots.length){
    timer.textContent='No solution found';
    solving=false;return;
  }

  const solved=snapshots[snapshots.length-1].solved;
  timer.textContent=`${snapshots.length} moves in ${elapsed}s (model inference)`;

  // Replay the solution visually
  const replayDelay=parseInt(document.getElementById('speed').value);
  for(let i=0;i<snapshots.length;i++){
    if(solveAbort)break;
    const snap=snapshots[i];
    updateCube(snap.face_grids);
    updateStats(snap);
    // Move tag
    const prev=movesList.querySelector('.latest');
    if(prev)prev.classList.remove('latest');
    const tag=document.createElement('span');
    tag.className='move-tag latest';tag.textContent=snap.move;
    movesList.appendChild(tag);movesList.scrollTop=movesList.scrollHeight;

    if(replayDelay>0&&i<snapshots.length-1){
      await new Promise(r=>setTimeout(r,replayDelay));
    }
  }
  if(solved){
    badge.className='status-badge solved';badge.textContent='Solved';
    document.getElementById('scene').classList.add('solved-glow');
  }
  solving=false;
}

async function doSolveBeam(){
  if(solving)return;solving=true;solveAbort=false;
  document.getElementById('btn-solve').disabled=true;
  document.getElementById('btn-beam').disabled=true;
  const badge=document.getElementById('status');
  badge.className='status-badge solving';badge.textContent='Beam search\u2026';
  const timer=document.getElementById('solve-timer');
  const movesList=document.getElementById('moves-list');
  movesList.innerHTML='';
  const t0=performance.now();

  // Start beam search on server
  await api('solve_beam',{beam_width:32,max_steps:100});

  // Poll for progress
  let done=false;
  while(!done&&!solveAbort){
    await new Promise(r=>setTimeout(r,300));
    const p=await api('beam_status');
    if(!p)continue;

    // Update live stats
    const elapsed=((performance.now()-t0)/1000).toFixed(1);
    timer.innerHTML=`<b>Step ${p.step||0}/${p.max_steps||100}</b> \u00b7 `+
      `${p.active_beams||0} beams \u00b7 `+
      `${p.states_explored||0} states explored \u00b7 `+
      `dist\u2248${p.best_value||'?'} \u00b7 `+
      `residual=${p.best_residual!=null?p.best_residual:'?'} \u00b7 `+
      `${elapsed}s`;

    // Show best beam's cube state live
    if(p.best_face_grids){
      updateCube(p.best_face_grids);
    }

    if(p.status==='done'||p.status==='error'){
      done=true;
      const snapshots=p.snapshots||[];
      if(!snapshots.length){
        timer.textContent='No solution found ('+elapsed+'s)';
        solving=false;return;
      }

      const solved=snapshots[snapshots.length-1].solved;
      const finalElapsed=((performance.now()-t0)/1000).toFixed(1);
      timer.textContent=(solved?'\u2705 Solved! ':'')+ snapshots.length+' moves, '+
        p.states_explored+' states explored in '+finalElapsed+'s';

      // Replay the winning solution
      // First reset cube display to scrambled state
      const resetData=await api('reset',{size:cubeSize});
      // Re-scramble with same scramble (state already has it)
      // Actually the snapshots already contain the face grids, just replay them
      const replayDelay=parseInt(document.getElementById('speed').value);
      for(let i=0;i<snapshots.length;i++){
        if(solveAbort)break;
        const snap=snapshots[i];
        updateCube(snap.face_grids);
        updateStats(snap);
        const prev=movesList.querySelector('.latest');
        if(prev)prev.classList.remove('latest');
        const tag=document.createElement('span');
        tag.className='move-tag latest';tag.textContent=snap.move;
        movesList.appendChild(tag);movesList.scrollTop=movesList.scrollHeight;
        if(replayDelay>0&&i<snapshots.length-1){
          await new Promise(r=>setTimeout(r,replayDelay));
        }
      }
      if(solved){
        badge.className='status-badge solved';badge.textContent='Solved';
        document.getElementById('scene').classList.add('solved-glow');
      }
    }
  }
  solving=false;
}

async function doReset(){
  solveAbort=true;solving=false;
  const data=await api('reset',{size:cubeSize});
  updateCube(data.face_grids);updateStats(data);
  document.getElementById('moves-list').innerHTML='';
  document.getElementById('solve-timer').textContent='';
  document.getElementById('btn-solve').disabled=true;
  document.getElementById('btn-beam').disabled=true;
}

document.getElementById('speed').addEventListener('input',e=>{
  document.getElementById('speed-val').textContent=e.target.value+'ms';
});
document.getElementById('scramble-depth').addEventListener('input',e=>{
  document.getElementById('scramble-depth-val').textContent=e.target.value;
});

// Drag to rotate 3D cube
let isDragging=false,prevX=0,prevY=0,rotX=-25,rotY=35;
const scene=document.getElementById('scene'),cube3d=document.getElementById('cube3d');
scene.addEventListener('mousedown',e=>{isDragging=true;prevX=e.clientX;prevY=e.clientY});
window.addEventListener('mousemove',e=>{
  if(!isDragging)return;
  rotY+=(e.clientX-prevX)*0.5;rotX-=(e.clientY-prevY)*0.5;
  cube3d.style.transform=`rotateX(${rotX}deg) rotateY(${rotY}deg)`;
  prevX=e.clientX;prevY=e.clientY;
});
window.addEventListener('mouseup',()=>{isDragging=false});

// Init
initStickers(2);
(async()=>{const data=await api('reset');updateCube(data.face_grids);updateStats(data)})();
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
            size = body.get('size', state.cube_size)
            state.scramble(length, size=size)
            result = state.to_json()

        elif self.path == '/api/step':
            move, solved = state.step()
            result = state.to_json()
            if move:
                result['move'] = f"{move.face} {move.turn_name()}"
            else:
                result['move'] = None

        elif self.path == '/api/solve_all':
            snapshots = state.solve_all()
            result = {
                "snapshots": snapshots,
                "final": state.to_json(),
            }

        elif self.path == '/api/solve_beam':
            beam_width = body.get('beam_width', 32)
            max_steps = body.get('max_steps', 100)
            state.start_beam_search(beam_width=beam_width, max_steps=max_steps)
            result = {"status": "started"}

        elif self.path == '/api/beam_status':
            p = getattr(state, 'beam_progress', None)
            if p is None:
                result = {"status": "idle"}
            else:
                result = dict(p)
                # Don't send snapshots until done (too large)
                if result["status"] != "done":
                    result.pop("snapshots", None)
                result["final"] = state.to_json()

        elif self.path == '/api/reset':
            size = body.get('size', state.cube_size)
            state.reset(size=size)
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
