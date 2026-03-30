"""
Training script for CayleyPy-style 4x4 distance prediction.

Trains a small ResMLP (~3M params) to predict random-walk distance
from any 4x4 cube state to the solved state. Uses online data generation
(no pre-computed dataset needed).

Usage: python train_distance_444.py
"""

import os
import sys
import time
import random

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from distance_444 import (
    DistanceMLP444,
    sample_walk_batch,
    encode_cube_444,
    evaluate_distance_search,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HIDDEN_DIM = 1024       # ResMLP hidden dimension
N_BLOCKS = 4            # number of residual blocks
EMBED_DIM = 32          # per-color embedding dimension
BATCH_SIZE = 4096       # training batch size (online generation)
MAX_WALK_DEPTH = 48     # max scramble depth for training walks
LR = 1e-3               # learning rate
WEIGHT_DECAY = 1e-5     # small weight decay
TIME_BUDGET = 14400     # 4 hours
EVAL_EVERY = 5000       # steps between smoke evals
EVAL_CUBES = 32         # cubes per smoke eval
EVAL_BEAM_WIDTH = 8     # beam width for smoke eval
CHECKPOINT_EVERY = 10000

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = DistanceMLP444(
    hidden_dim=HIDDEN_DIM, n_blocks=N_BLOCKS, embed_dim=EMBED_DIM
).to(device)
print(f"Model: {model.num_params():,} parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=TIME_BUDGET // 1  # approximate steps
)

# Create run directory
from datetime import datetime
from pathlib import Path
run_dir = Path("runs") / f"dist444_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_dir.mkdir(parents=True, exist_ok=True)
print(f"Run dir: {run_dir}")

# wandb
try:
    import wandb
    wandb.init(
        project="rubiks-cube-solver",
        name=f"dist444_{run_dir.name}",
        config={
            "model": "DistanceMLP444",
            "hidden_dim": HIDDEN_DIM,
            "n_blocks": N_BLOCKS,
            "embed_dim": EMBED_DIM,
            "batch_size": BATCH_SIZE,
            "max_walk_depth": MAX_WALK_DEPTH,
            "lr": LR,
            "time_budget": TIME_BUDGET,
            "num_params": model.num_params(),
        },
    )
    use_wandb = True
except Exception:
    use_wandb = False
    print("wandb not available, logging to stdout only")

# Load eval episodes
try:
    from prepare import load_dataset
    from rubiks import Episode
    payload = load_dataset()
    eval_eps = payload.get("eval_episodes", {}).get("id", [])
    eval_4x4 = [e for e in eval_eps if (Episode.from_dict(e) if isinstance(e, dict) else e).size == 4]
    if not eval_4x4:
        # Try OOD
        eval_eps_ood = payload.get("eval_episodes", {}).get("ood_dev", [])
        eval_4x4 = [e for e in eval_eps_ood if (Episode.from_dict(e) if isinstance(e, dict) else e).size == 4]
    print(f"Eval episodes: {len(eval_4x4)} 4x4 cubes")
except Exception as e:
    print(f"Could not load eval episodes: {e}")
    eval_4x4 = []

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

rng = random.Random(42)
t_start = time.time()
step = 0
smooth_loss = 0.0
best_stage = 0.0

print(f"\nTraining for {TIME_BUDGET}s...")
print(f"Batch size: {BATCH_SIZE}, Max depth: {MAX_WALK_DEPTH}")
print()

while True:
    t0 = time.time()

    # Generate training batch online
    samples = sample_walk_batch(BATCH_SIZE // MAX_WALK_DEPTH + 1, MAX_WALK_DEPTH, rng)

    # Collate
    states = torch.stack([s for s, _ in samples]).to(device)
    distances = torch.tensor([d for _, d in samples], dtype=torch.float32).to(device)

    # Forward + loss
    pred = model(states)
    loss = F.smooth_l1_loss(pred, distances)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    dt = time.time() - t0
    elapsed = time.time() - t_start
    remaining = TIME_BUDGET - elapsed

    # Logging
    ema_beta = 0.95
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss.item()
    debiased = smooth_loss / (1 - ema_beta ** (step + 1))

    if step % 100 == 0:
        samples_per_sec = len(samples) / dt
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\rstep {step:06d} | loss: {debiased:.4f} | lr: {lr_now:.6f} | "
              f"samples/s: {samples_per_sec:.0f} | remaining: {remaining:.0f}s    ",
              end="", flush=True)

        if use_wandb and step % 500 == 0:
            wandb.log({
                "train/loss": debiased,
                "train/lr": lr_now,
                "train/samples_per_sec": samples_per_sec,
            }, step=step)

    # Smoke eval
    if step > 0 and step % EVAL_EVERY == 0 and eval_4x4:
        print(f"\n  Smoke eval (beam w={EVAL_BEAM_WIDTH}, {EVAL_CUBES} cubes)...", end="", flush=True)
        model.eval()
        metrics = evaluate_distance_search(
            model, eval_4x4, beam_width=EVAL_BEAM_WIDTH,
            max_steps=100, device=device, num_cubes=EVAL_CUBES,
        )
        model.train()

        print(f" stage={metrics['mean_stage_reached']:.2f} | "
              f"centers={metrics['centers_done_rate']:.0%} | "
              f"edges={metrics['edges_paired_rate']:.0%} | "
              f"reduced={metrics['reduced_to_3x3_rate']:.0%} | "
              f"solved={metrics['solve_rate']:.0%}")

        if use_wandb:
            wandb.log({
                "eval/mean_stage_reached": metrics["mean_stage_reached"],
                "eval/centers_done_rate": metrics["centers_done_rate"],
                "eval/edges_paired_rate": metrics["edges_paired_rate"],
                "eval/reduced_to_3x3_rate": metrics["reduced_to_3x3_rate"],
                "eval/solve_rate": metrics["solve_rate"],
            }, step=step)

        if metrics["mean_stage_reached"] > best_stage:
            best_stage = metrics["mean_stage_reached"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "step": step,
                "loss": debiased,
                "metrics": metrics,
            }, run_dir / "model_best.pt")
            print(f"  [BEST stage={best_stage:.2f}, saved]")

    # Checkpoint
    if step > 0 and step % CHECKPOINT_EVERY == 0:
        torch.save({
            "model_state_dict": model.state_dict(),
            "step": step,
            "loss": debiased,
        }, run_dir / "model_latest.pt")

    step += 1

    # Time's up
    if elapsed >= TIME_BUDGET:
        break

# Save final checkpoint
torch.save({
    "model_state_dict": model.state_dict(),
    "step": step,
    "loss": debiased,
}, run_dir / "model_final.pt")

print(f"\n\nTraining complete: {step} steps in {elapsed:.0f}s")
print(f"Best stage reached: {best_stage:.2f}")
print(f"Final loss: {debiased:.4f}")
print(f"Checkpoints: {run_dir}")

# Final eval
if eval_4x4:
    print(f"\nFinal eval (beam w=32, 64 cubes)...")
    model.eval()
    final_metrics = evaluate_distance_search(
        model, eval_4x4, beam_width=32,
        max_steps=200, device=device, num_cubes=64,
    )
    print(f"  solve_rate: {final_metrics['solve_rate']:.0%}")
    print(f"  centers_done_rate: {final_metrics['centers_done_rate']:.0%}")
    print(f"  edges_paired_rate: {final_metrics['edges_paired_rate']:.0%}")
    print(f"  reduced_to_3x3_rate: {final_metrics['reduced_to_3x3_rate']:.0%}")
    print(f"  mean_stage_reached: {final_metrics['mean_stage_reached']:.2f}")

    if use_wandb:
        wandb.log({
            "final/solve_rate": final_metrics["solve_rate"],
            "final/centers_done_rate": final_metrics["centers_done_rate"],
            "final/edges_paired_rate": final_metrics["edges_paired_rate"],
            "final/reduced_to_3x3_rate": final_metrics["reduced_to_3x3_rate"],
            "final/mean_stage_reached": final_metrics["mean_stage_reached"],
        })

if use_wandb:
    wandb.finish()
