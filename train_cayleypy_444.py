"""
4x4 Rubik's cube solver using CayleyPy's infrastructure.

Uses CayleyPy for:
- Fast random walk generation (native, ~21K states/sec)
- Beam search with learned heuristic
- State encoding/manipulation

We train a small ResMLP to predict walk distance, then use it as
the beam search heuristic via CayleyPy's Predictor interface.
"""

import os, sys, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import cayleypy
from cayleypy.algo import BeamSearchAlgorithm

# ---------------------------------------------------------------------------
# Setup CayleyPy graph
# ---------------------------------------------------------------------------

print("Creating 4x4 CayleyPy graph...")
graph_def = cayleypy.Puzzles.rubik_cube(4, 'HTM')
graph = cayleypy.CayleyGraph(graph_def)
print(f"Graph: {graph_def.n_generators} generators, state_size={graph_def.state_size}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    def forward(self, x):
        return x + self.net(x)

class DistanceNet(nn.Module):
    def __init__(self, state_size=96, n_colors=6, embed_dim=32, hidden=1024, blocks=4):
        super().__init__()
        self.embed = nn.Embedding(n_colors, embed_dim)
        self.proj = nn.Linear(state_size * embed_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(blocks)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, 96) long tensor of color indices
        e = self.embed(x).view(x.size(0), -1)
        h = F.relu(self.proj(e))
        for b in self.blocks:
            h = F.relu(b(h))
        return self.head(h).squeeze(-1)

# ---------------------------------------------------------------------------
# CayleyPy Predictor wrapper
# ---------------------------------------------------------------------------

class TorchPredictor:
    """Wraps our PyTorch model as a CayleyPy-compatible predictor."""
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def __call__(self, states):
        # states: torch.Tensor of shape (N, 96)
        with torch.no_grad():
            if states.device != torch.device(self.device):
                states = states.to(self.device)
            return self.model(states.long()).cpu()

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

HIDDEN = 1024
BLOCKS = 4
EMBED = 32
BATCH_WALKS = 200       # walks per batch
WALK_LENGTH = 48        # max walk depth
LR = 1e-3
TIME_BUDGET = 14400     # 4 hours
EVAL_EVERY = 2000
EVAL_CUBES = 16
EVAL_BEAM = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DistanceNet(hidden=HIDDEN, blocks=BLOCKS, embed_dim=EMBED).to(device)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

# wandb
try:
    import wandb
    wandb.init(project="rubiks-cube-solver", name="cayleypy_444",
               config={"hidden": HIDDEN, "blocks": BLOCKS, "embed": EMBED,
                       "batch_walks": BATCH_WALKS, "walk_length": WALK_LENGTH,
                       "lr": LR, "time_budget": TIME_BUDGET})
    use_wandb = True
except:
    use_wandb = False

# Run dir
from pathlib import Path
from datetime import datetime
run_dir = Path("runs") / f"cayleypy444_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_dir.mkdir(parents=True, exist_ok=True)
print(f"Run dir: {run_dir}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start = time.time()
step = 0
smooth_loss = 0.0
best_solved = -1

print(f"\nTraining for {TIME_BUDGET}s with CayleyPy random walks...")
print(f"Walks/batch: {BATCH_WALKS}, Walk length: {WALK_LENGTH}")
print()

while True:
    t0 = time.time()

    # Generate training data using CayleyPy (FAST)
    x, y = graph.random_walks(width=BATCH_WALKS, length=WALK_LENGTH, mode='classic')
    # x: (BATCH_WALKS * WALK_LENGTH, 96) states
    # y: (BATCH_WALKS * WALK_LENGTH,) distances

    x_gpu = x.to(device).long()
    y_gpu = y.to(device).float()

    # Forward + loss
    pred = model(x_gpu)
    loss = F.smooth_l1_loss(pred, y_gpu)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    dt = time.time() - t0
    elapsed = time.time() - t_start

    ema = 0.95
    smooth_loss = ema * smooth_loss + (1 - ema) * loss.item()
    debiased = smooth_loss / (1 - ema ** (step + 1))

    if step % 100 == 0:
        remaining = TIME_BUDGET - elapsed
        print(f"\rstep {step:06d} | loss: {debiased:.4f} | dt: {dt*1000:.0f}ms | "
              f"states/step: {x.shape[0]} | remaining: {remaining:.0f}s    ",
              end="", flush=True)
        if use_wandb and step % 500 == 0:
            wandb.log({"train/loss": debiased}, step=step)

    # Eval: try beam search on scrambled cubes
    if step > 0 and step % EVAL_EVERY == 0:
        print(f"\n  Eval (beam w={EVAL_BEAM}, {EVAL_CUBES} cubes)...", end="", flush=True)
        model.eval()
        predictor = TorchPredictor(model, device)
        bs = BeamSearchAlgorithm(graph)

        solved = 0
        for i in range(EVAL_CUBES):
            # Generate a random scramble
            scramble_x, scramble_y = graph.random_walks(width=1, length=WALK_LENGTH, mode='classic')
            start_state = scramble_x[-1]  # last state of the walk (most scrambled)

            result = bs.search(
                start_state=start_state,
                predictor=predictor,
                beam_width=EVAL_BEAM,
                max_steps=200,
                beam_mode='advanced',
                history_depth=2,
            )
            if result.path_found:
                solved += 1

        model.train()
        sr = solved / EVAL_CUBES
        print(f" solved: {solved}/{EVAL_CUBES} ({sr:.0%})")

        if use_wandb:
            wandb.log({"eval/solve_rate": sr, "eval/solved": solved}, step=step)

        if solved > best_solved:
            best_solved = solved
            torch.save({"model_state_dict": model.state_dict(), "step": step,
                        "loss": debiased, "solved": solved},
                       run_dir / "model_best.pt")
            print(f"  [BEST solved={solved}]")

    # Checkpoint
    if step > 0 and step % 10000 == 0:
        torch.save({"model_state_dict": model.state_dict(), "step": step,
                    "loss": debiased}, run_dir / "model_latest.pt")

    step += 1
    if elapsed >= TIME_BUDGET:
        break

print(f"\n\nDone: {step} steps, {elapsed:.0f}s, best_solved={best_solved}/{EVAL_CUBES}")
torch.save({"model_state_dict": model.state_dict(), "step": step, "loss": debiased},
           run_dir / "model_final.pt")

# Final eval with wider beam
print(f"\nFinal eval (beam w=1000, {EVAL_CUBES} cubes)...")
model.eval()
predictor = TorchPredictor(model, device)
bs = BeamSearchAlgorithm(graph)
solved = 0
for i in range(EVAL_CUBES):
    scramble_x, scramble_y = graph.random_walks(width=1, length=WALK_LENGTH, mode='classic')
    start_state = scramble_x[-1]
    result = bs.search(start_state=start_state, predictor=predictor,
                       beam_width=1000, max_steps=200, beam_mode='advanced', history_depth=2)
    if result.path_found:
        solved += 1
print(f"Final: {solved}/{EVAL_CUBES} ({solved/EVAL_CUBES:.0%})")

if use_wandb:
    wandb.log({"final/solve_rate": solved / EVAL_CUBES})
    wandb.finish()
