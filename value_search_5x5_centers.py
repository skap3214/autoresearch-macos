"""
DeepCubeA-style value network + beam search for 5x5 Rubik's cube CENTER solving.

Key insight: instead of predicting walk depth (which failed for 4x4), we predict
the COUNT OF UNSOLVED CENTER STICKERS (0-54). This gives a dense, meaningful
reward signal since centers break gradually under random moves.

Training is fully self-supervised: scramble from solved, count wrong stickers.

Usage:
    python value_search_5x5_centers.py --hours 4
    python value_search_5x5_centers.py --eval-only --checkpoint checkpoints/value_5x5_centers/final.pt

Based on: Agostinelli et al., "Solving the Rubik's Cube with Deep Reinforcement
Learning and Search" (Nature Machine Intelligence, 2019)
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rubiks import Cube, Move, FACE_ORDER, FACE_COLORS

# Import from solve_5x5_teacher
from solve_5x5_teacher import (
    centers_done_555,
    extract_center_stickers,
    ALL_MOVES_5x5,
    COLOR_TO_IDX,
    NUM_COLORS,
    NUM_CENTER_STICKERS,
    generate_scramble,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ACTIONS = len(ALL_MOVES_5x5)  # 36

# Precompute inverse move indices for pruning
_INVERSE_MOVE_IDX: dict[int, int] = {}
for _i, _m in enumerate(ALL_MOVES_5x5):
    _inv = _m.inverse()
    for _j, _m2 in enumerate(ALL_MOVES_5x5):
        if _m2.face == _inv.face and _m2.width == _inv.width and _m2.turns == _inv.turns:
            _INVERSE_MOVE_IDX[_i] = _j
            break


# ---------------------------------------------------------------------------
# Target computation: count unsolved center stickers
# ---------------------------------------------------------------------------

def count_unsolved_centers(cube: Cube) -> int:
    """Count the number of center stickers that differ from the face majority.

    For each face, the 9 center stickers should all be the same color when
    centers are solved. We count how many stickers differ from the most
    common color on each face (which handles partial solving correctly).

    Returns: integer in [0, 54]. 0 means all centers solved.
    """
    total_wrong = 0
    for face in FACE_ORDER:
        grid = cube.face_grid(face)
        colors = []
        for r in range(1, 4):
            for c in range(1, 4):
                colors.append(grid[r][c])
        # Count most common color
        from collections import Counter
        counts = Counter(colors)
        most_common_count = counts.most_common(1)[0][1]
        total_wrong += 9 - most_common_count
    return total_wrong


def count_unsolved_centers_canonical(cube: Cube) -> int:
    """Count center stickers that are wrong relative to the center-of-center.

    Uses grid[2][2] (the fixed center) as the canonical color for each face.
    This is more stable than majority voting.

    Returns: integer in [0, 48]. 0 means all centers solved.
    (Max is 48, not 54, because the 6 center-of-centers are always 'correct'.)
    """
    total_wrong = 0
    for face in FACE_ORDER:
        grid = cube.face_grid(face)
        canonical = grid[2][2]  # center of center (fixed piece)
        for r in range(1, 4):
            for c in range(1, 4):
                if (r, c) == (2, 2):
                    continue  # skip center-of-center
                if grid[r][c] != canonical:
                    total_wrong += 1
    return total_wrong


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def encode_centers(cube: Cube) -> list[int]:
    """Extract 54 center sticker color indices from a 5x5 cube."""
    return extract_center_stickers(cube)


def encode_centers_batch(cubes: list[Cube]) -> torch.LongTensor:
    """Encode a batch of cubes' center stickers. Returns (B, 54) tensor."""
    batch = [extract_center_stickers(c) for c in cubes]
    return torch.tensor(batch, dtype=torch.long)


# ---------------------------------------------------------------------------
# Move pruning
# ---------------------------------------------------------------------------

def get_valid_move_indices(last_move_idx: Optional[int]) -> list[int]:
    """Return indices of valid moves given the last move index.
    Prunes inverse moves to avoid immediate backtracking."""
    if last_move_idx is None:
        return list(range(N_ACTIONS))
    inv_idx = _INVERSE_MOVE_IDX.get(last_move_idx)
    valid = []
    for i in range(N_ACTIONS):
        if i == inv_idx:
            continue
        valid.append(i)
    return valid


# ---------------------------------------------------------------------------
# Model: MLP-Mixer for center distance prediction
# ---------------------------------------------------------------------------

class MLPBlock(nn.Module):
    """Two-layer MLP with GELU."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class MixerLayer(nn.Module):
    """MLP-Mixer layer: token-mixing + channel-mixing."""
    def __init__(self, n_tokens: int, d_model: int, token_mix_dim: int,
                 channel_mix_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.token_mix = MLPBlock(n_tokens, token_mix_dim, n_tokens, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mix = MLPBlock(d_model, channel_mix_dim, d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        y = self.norm1(x)
        y = y.transpose(1, 2)  # (B, D, S)
        y = self.token_mix(y)  # (B, D, S)
        y = y.transpose(1, 2)  # (B, S, D)
        x = x + y
        y = self.norm2(x)
        y = self.channel_mix(y)
        x = x + y
        return x


class CenterDistanceNet(nn.Module):
    """Per-sticker token encoder + MLP-Mixer for predicting unsolved center count.

    Each of 54 center stickers gets:
      - color embedding (6 colors)
      - face embedding (6 faces)
      - position features (row, col within the 3x3 center grid)

    Then N MLP-Mixer layers, global average pool, scalar output.
    Target: ~5-10M parameters.
    """

    def __init__(
        self,
        n_stickers: int = 54,
        n_colors: int = 6,
        n_faces: int = 6,
        color_embed_dim: int = 64,
        face_embed_dim: int = 32,
        pos_dim: int = 8,
        d_model: int = 256,
        n_layers: int = 6,
        token_mix_dim: int = 128,
        channel_mix_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_stickers = n_stickers
        self.d_model = d_model

        self.color_embed = nn.Embedding(n_colors, color_embed_dim)
        self.face_embed = nn.Embedding(n_faces, face_embed_dim)

        # Position encoding: row and col -> small projection
        self.pos_proj = nn.Linear(2, pos_dim)

        # Input projection: concat embeddings -> d_model
        input_dim = color_embed_dim + face_embed_dim + pos_dim
        self.input_proj = nn.Linear(input_dim, d_model)

        # Mixer layers
        self.mixer_layers = nn.ModuleList([
            MixerLayer(n_stickers, d_model, token_mix_dim, channel_mix_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Distance head: predicts count of unsolved center stickers
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Precompute face indices and position features for 54 stickers
        # 9 stickers per face, 6 faces, in the order of FACE_ORDER
        face_indices = []
        pos_features = []
        for face_idx in range(6):
            for r in range(3):
                for c in range(3):
                    face_indices.append(face_idx)
                    # Normalize row, col to [-1, 1]
                    pos_features.append([r / 1.0 - 1.0, c / 1.0 - 1.0])
        self.register_buffer("face_indices", torch.tensor(face_indices, dtype=torch.long))
        self.register_buffer("pos_features", torch.tensor(pos_features, dtype=torch.float32))

    def forward(self, stickers: torch.Tensor) -> torch.Tensor:
        """stickers: (B, 54) integer tensor of color indices 0-5.
        Returns: (B,) predicted unsolved center count."""
        B = stickers.size(0)

        # Color embedding: (B, 54, color_embed_dim)
        color_emb = self.color_embed(stickers)

        # Face embedding: (B, 54, face_embed_dim)
        face_emb = self.face_embed(self.face_indices).unsqueeze(0).expand(B, -1, -1)

        # Position features: (B, 54, pos_dim)
        pos_emb = self.pos_proj(self.pos_features).unsqueeze(0).expand(B, -1, -1)

        # Concatenate and project
        x = torch.cat([color_emb, face_emb, pos_emb], dim=-1)
        x = self.input_proj(x)  # (B, 54, d_model)

        # Mixer layers
        for layer in self.mixer_layers:
            x = layer(x)
        x = self.final_norm(x)

        # Global average pooling
        x = x.mean(dim=1)  # (B, d_model)

        # Predict unsolved count (clamped to [0, 54])
        out = self.head(x).squeeze(-1)  # (B,)
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Self-supervised data generation: count unsolved center stickers
# ---------------------------------------------------------------------------

def generate_training_batch(
    batch_size: int,
    max_depth: int,
    rng: random.Random,
    include_solved: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate training data via random walks from solved state.

    For each sample:
      1. Start from solved 5x5 cube
      2. Apply d random moves (d ~ Uniform[1, max_depth])
      3. Count unsolved center stickers as target

    KEY: target is count of wrong stickers (0-48), NOT walk depth.
    This gives a dense, meaningful signal.

    Returns: (states, targets) tensors.
    """
    all_indices = []
    all_targets = []

    if include_solved:
        cube = Cube(5)
        all_indices.append(extract_center_stickers(cube))
        all_targets.append(0.0)

    for _ in range(batch_size):
        d = rng.randint(1, max_depth)
        cube = Cube(5)

        for step in range(d):
            move = ALL_MOVES_5x5[rng.randint(0, N_ACTIONS - 1)]
            cube.apply_move(move)

        stickers = extract_center_stickers(cube)
        target = float(count_unsolved_centers_canonical(cube))

        all_indices.append(stickers)
        all_targets.append(target)

    states = torch.tensor(all_indices, dtype=torch.long)
    targets = torch.tensor(all_targets, dtype=torch.float32)
    return states, targets


# ---------------------------------------------------------------------------
# DeepCubeA-style bootstrap: 1 + min(children) but with unsolved-count twist
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_bootstrap_targets(
    model: CenterDistanceNet,
    batch_size: int,
    max_depth: int,
    rng: random.Random,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate bootstrap training targets.

    For each scrambled state:
      - If centers solved: target = 0
      - Else: target = min over children of model(child)
        (No +1 because we predict sticker count, not step count.
         The minimum child value is the best reachable state.)

    This bootstraps the heuristic toward the true optimal unsolved count
    reachable in one move.
    """
    model.eval()

    # Generate scrambled states
    cubes = []
    for _ in range(batch_size):
        d = rng.randint(1, max_depth)
        cube = Cube(5)
        for step in range(d):
            move = ALL_MOVES_5x5[rng.randint(0, N_ACTIONS - 1)]
            cube.apply_move(move)
        cubes.append(cube)

    # Include some solved states
    n_solved = max(1, batch_size // 20)
    for _ in range(n_solved):
        cubes.append(Cube(5))

    B = len(cubes)
    all_states = encode_centers_batch(cubes).to(device)

    # Expand all 36 children for each state
    child_cubes = []
    for cube in cubes:
        for move in ALL_MOVES_5x5:
            child = cube.copy()
            child.apply_move(move)
            child_cubes.append(child)

    child_states = encode_centers_batch(child_cubes).to(device)

    # Score children in chunks
    chunk_size = 2048
    all_scores = []
    for i in range(0, len(child_cubes), chunk_size):
        chunk = child_states[i:i + chunk_size]
        scores = model(chunk).cpu()
        all_scores.append(scores)
    child_values = torch.cat(all_scores, dim=0).view(B, N_ACTIONS)

    # Target = min over children (best reachable in one move)
    targets = child_values.min(dim=1).values

    # Override: solved states get target = 0
    for i, cube in enumerate(cubes):
        if centers_done_555(cube):
            targets[i] = 0.0

    targets = targets.clamp(min=0.0)

    return all_states, targets.to(device)


# ---------------------------------------------------------------------------
# Beam search solver
# ---------------------------------------------------------------------------

@dataclass
class BeamItem:
    cube: Cube
    history: list[int]  # list of move indices
    last_move_idx: Optional[int]


def _center_state_key(cube: Cube) -> str:
    """Hash key for center stickers only (ignoring edges/corners)."""
    stickers = extract_center_stickers(cube)
    return "".join(str(s) for s in stickers)


@torch.no_grad()
def beam_search_5x5_centers(
    model: CenterDistanceNet,
    cube: Cube,
    beam_width: int = 64,
    max_steps: int = 300,
    device: str = "cuda",
) -> tuple[bool, list[Move], int]:
    """Beam search using unsolved-center-count network as heuristic.

    Returns (solved, move_history, nodes_expanded).
    """
    model.eval()

    if centers_done_555(cube):
        return True, [], 0

    beam = [BeamItem(cube=cube.copy(), history=[], last_move_idx=None)]
    visited: set[str] = {_center_state_key(cube)}
    nodes_expanded = 0

    for step in range(max_steps):
        # Expand all beam items
        candidates: list[tuple[Cube, list[int], int, str]] = []
        for item in beam:
            if centers_done_555(item.cube):
                moves = [ALL_MOVES_5x5[i] for i in item.history]
                return True, moves, nodes_expanded

            valid_moves = get_valid_move_indices(item.last_move_idx)
            for move_idx in valid_moves:
                move = ALL_MOVES_5x5[move_idx]
                new_cube = item.cube.copy()
                new_cube.apply_move(move)
                state_key = _center_state_key(new_cube)
                if state_key in visited:
                    continue
                candidates.append((
                    new_cube,
                    item.history + [move_idx],
                    move_idx,
                    state_key,
                ))

        if not candidates:
            break

        nodes_expanded += len(candidates)

        # Score all candidates with distance network
        states = encode_centers_batch([c for c, _, _, _ in candidates]).to(device)
        # Process in chunks to avoid OOM
        chunk_size = 4096
        all_scores = []
        for i in range(0, len(candidates), chunk_size):
            chunk = states[i:i + chunk_size]
            scores = model(chunk).cpu()
            all_scores.append(scores)
        pred_distances = torch.cat(all_scores, dim=0)

        # Sort by predicted unsolved count (lower = better)
        scored = []
        for i, (c, hist, last_idx, key) in enumerate(candidates):
            h = pred_distances[i].item()
            scored.append((h, i, c, hist, last_idx, key))

        scored.sort(key=lambda x: x[0])

        # Check for solved and build new beam
        beam = []
        for h, i, c, hist, last_idx, key in scored:
            if centers_done_555(c):
                moves = [ALL_MOVES_5x5[mi] for mi in hist]
                return True, moves, nodes_expanded
            if len(beam) >= beam_width:
                continue
            if key not in visited:
                visited.add(key)
                beam.append(BeamItem(cube=c, history=hist, last_move_idx=last_idx))

        if not beam:
            break

    return False, [], nodes_expanded


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_solver(
    model: CenterDistanceNet,
    num_cubes: int = 50,
    scramble_length: int = 30,
    beam_width: int = 64,
    max_steps: int = 300,
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Evaluate beam search on random 5x5 scrambles (center solving).

    Returns dict with solve_rate, mean_solution_length, mean_nodes, etc.
    """
    rng = random.Random(seed)
    solved_count = 0
    total_solution_len = 0
    total_nodes = 0
    unsolved_remaining = []

    for i in range(num_cubes):
        cube = Cube(5)
        scramble = generate_scramble(rng, scramble_length)
        cube.apply_moves(scramble)

        if centers_done_555(cube):
            solved_count += 1
            total_solution_len += 0
            continue

        solved, history, nodes = beam_search_5x5_centers(
            model, cube, beam_width=beam_width, max_steps=max_steps, device=device
        )

        if solved:
            solved_count += 1
            total_solution_len += len(history)
        else:
            # Check how many stickers remain unsolved
            # Replay the last beam's best state isn't stored, so just note failure
            unsolved_remaining.append(count_unsolved_centers_canonical(cube))
        total_nodes += nodes

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{num_cubes}] solved={solved_count}/{i+1}")

    n = num_cubes
    return {
        "solve_rate": solved_count / n,
        "solved": solved_count,
        "total": n,
        "mean_solution_length": total_solution_len / max(1, solved_count),
        "mean_nodes_expanded": total_nodes / n,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    """Main training loop."""
    try:
        import wandb
        use_wandb = True
    except ImportError:
        use_wandb = False
        print("wandb not available, logging to stdout only")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, falling back to CPU")

    # Create model
    model = CenterDistanceNet(
        d_model=args.d_model,
        n_layers=args.n_layers,
        token_mix_dim=args.token_mix_dim,
        channel_mix_dim=args.channel_mix_dim,
        dropout=args.dropout,
    ).to(device)

    print(f"Model parameters: {model.num_params():,}", flush=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Cosine annealing with warmup
    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, args.total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Wandb
    if use_wandb:
        wandb.init(
            project="rubiks-value-search-5x5-centers",
            config=vars(args),
            name=f"centers-5x5-{args.d_model}d-{args.n_layers}L",
        )

    # Checkpoint directory
    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    rng = random.Random(args.seed)
    start_time = time.time()
    max_seconds = args.hours * 3600
    global_step = 0

    # Curriculum: start with short walks, gradually increase
    curr_max_depth = args.curriculum_start

    # Phase tracking
    use_bootstrap = False
    bootstrap_start_step = args.bootstrap_after

    print(f"Training for up to {args.hours} hours ({max_seconds:.0f}s)", flush=True)
    print(f"Phase 1: Supervised sticker-count (steps 0-{bootstrap_start_step})")
    print(f"Phase 2: Bootstrap min-child (steps {bootstrap_start_step}+)")
    print(f"Curriculum: depth {args.curriculum_start} -> {args.curriculum_end}")
    print(f"Target: count of unsolved center stickers (0-48)")
    print(flush=True)

    # Load checkpoint if exists
    latest_ckpt = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["step"]
        curr_max_depth = ckpt.get("curr_max_depth", args.curriculum_start)
        print(f"Resumed from checkpoint at step {global_step}")

    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            print(f"Time budget exhausted ({args.hours}h)")
            break
        if global_step >= args.total_steps:
            print(f"Reached max steps ({args.total_steps})")
            break

        model.train()

        # Curriculum: linearly increase max_depth
        progress = min(1.0, global_step / max(1, args.curriculum_ramp_steps))
        curr_max_depth = int(
            args.curriculum_start + progress * (args.curriculum_end - args.curriculum_start)
        )
        curr_max_depth = max(args.curriculum_start, min(args.curriculum_end, curr_max_depth))

        # Decide training mode
        use_bootstrap = global_step >= bootstrap_start_step

        if use_bootstrap:
            # Bootstrap: target = min over children
            states, targets = generate_bootstrap_targets(
                model, args.batch_size, curr_max_depth, rng, device
            )
            model.train()
        else:
            # Phase 1: supervised unsolved-center count
            states, targets = generate_training_batch(
                args.batch_size, curr_max_depth, rng
            )
            states = states.to(device)
            targets = targets.to(device)

        # Forward pass
        predictions = model(states)
        loss = F.smooth_l1_loss(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        global_step += 1

        # Logging
        if global_step % args.log_interval == 0:
            mae = (predictions - targets).abs().mean().item()
            lr_now = scheduler.get_last_lr()[0]
            phase = "bootstrap" if use_bootstrap else "supervised"
            log_dict = {
                "train/loss": loss.item(),
                "train/mae": mae,
                "train/lr": lr_now,
                "train/max_depth": curr_max_depth,
                "train/phase": 2 if use_bootstrap else 1,
                "train/step": global_step,
            }
            print(
                f"[{global_step:7d}] loss={loss.item():.4f} mae={mae:.3f} "
                f"lr={lr_now:.2e} depth={curr_max_depth} phase={phase} "
                f"elapsed={elapsed/3600:.2f}h",
                flush=True,
            )
            if use_wandb:
                wandb.log(log_dict, step=global_step)

        # Periodic evaluation
        if global_step % args.eval_interval == 0:
            print(f"\n--- Evaluation at step {global_step} ---")

            # Quick sanity check: prediction on solved state
            solved_cube = Cube(5)
            solved_stickers = torch.tensor(
                [extract_center_stickers(solved_cube)], dtype=torch.long, device=device
            )
            model.eval()
            solved_pred = model(solved_stickers).item()
            print(f"  Solved state prediction: {solved_pred:.2f} (should be ~0)")

            # Quick check: heavily scrambled state
            test_cube = Cube(5)
            test_scramble = generate_scramble(rng, 50)
            test_cube.apply_moves(test_scramble)
            test_stickers = torch.tensor(
                [extract_center_stickers(test_cube)], dtype=torch.long, device=device
            )
            scrambled_pred = model(test_stickers).item()
            actual_wrong = count_unsolved_centers_canonical(test_cube)
            print(f"  Scrambled state: pred={scrambled_pred:.2f} actual={actual_wrong}")

            for bw in args.eval_beam_widths:
                results = evaluate_solver(
                    model,
                    num_cubes=args.eval_num_cubes,
                    scramble_length=args.eval_scramble_length,
                    beam_width=bw,
                    max_steps=args.eval_max_steps,
                    device=device,
                    seed=args.eval_seed,
                    verbose=True,
                )
                print(
                    f"  beam_width={bw:4d}: solve_rate={results['solve_rate']:.3f} "
                    f"({results['solved']}/{results['total']}) "
                    f"mean_sol_len={results['mean_solution_length']:.1f} "
                    f"mean_nodes={results['mean_nodes_expanded']:.0f}"
                )
                if use_wandb:
                    wandb.log({
                        f"eval/solve_rate_bw{bw}": results["solve_rate"],
                        f"eval/mean_sol_len_bw{bw}": results["mean_solution_length"],
                        f"eval/mean_nodes_bw{bw}": results["mean_nodes_expanded"],
                    }, step=global_step)
            print(flush=True)

        # Checkpoint saving
        if global_step % args.save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
            ckpt_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": global_step,
                "curr_max_depth": curr_max_depth,
                "args": vars(args),
            }
            torch.save(ckpt_data, ckpt_path)
            torch.save(ckpt_data, latest_ckpt)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = os.path.join(ckpt_dir, "final.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": global_step,
        "curr_max_depth": curr_max_depth,
        "args": vars(args),
    }, final_path)
    print(f"Saved final checkpoint: {final_path}")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    for bw in [1, 16, 64, 128]:
        results = evaluate_solver(
            model,
            num_cubes=100,
            scramble_length=30,
            beam_width=bw,
            max_steps=300,
            device=device,
            seed=42,
            verbose=True,
        )
        print(
            f"  beam_width={bw:4d}: solve_rate={results['solve_rate']:.3f} "
            f"({results['solved']}/{results['total']}) "
            f"mean_sol_len={results['mean_solution_length']:.1f}"
        )

    if use_wandb:
        wandb.finish()

    return model


# ---------------------------------------------------------------------------
# Eval-only mode
# ---------------------------------------------------------------------------

def eval_only(args):
    """Load a checkpoint and run evaluation."""
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = CenterDistanceNet(
        d_model=args.d_model,
        n_layers=args.n_layers,
        token_mix_dim=args.token_mix_dim,
        channel_mix_dim=args.channel_mix_dim,
        dropout=0.0,  # no dropout at eval
    ).to(device)

    ckpt_path = args.checkpoint
    if not ckpt_path:
        ckpt_path = os.path.join(args.checkpoint_dir, "final.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")

    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint found at {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from step {ckpt['step']}")
    print(f"Model parameters: {model.num_params():,}")

    model.eval()

    # Sanity checks
    solved_cube = Cube(5)
    solved_stickers = torch.tensor(
        [extract_center_stickers(solved_cube)], dtype=torch.long, device=device
    )
    solved_pred = model(solved_stickers).item()
    print(f"Solved state prediction: {solved_pred:.2f} (should be ~0)")

    print(f"\n=== Evaluation ===")
    for bw in [1, 16, 64, 128, 256]:
        results = evaluate_solver(
            model,
            num_cubes=100,
            scramble_length=30,
            beam_width=bw,
            max_steps=300,
            device=device,
            seed=42,
            verbose=True,
        )
        print(
            f"  beam_width={bw:4d}: solve_rate={results['solve_rate']:.3f} "
            f"({results['solved']}/{results['total']}) "
            f"mean_sol_len={results['mean_solution_length']:.1f} "
            f"mean_nodes={results['mean_nodes_expanded']:.0f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepCubeA-style value+search for 5x5 center solving"
    )

    # Mode
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation (no training)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for eval-only mode")

    # Time budget
    parser.add_argument("--hours", type=float, default=4.0,
                        help="Training time budget in hours")

    # Model architecture
    parser.add_argument("--d_model", type=int, default=320,
                        help="Model hidden dimension")
    parser.add_argument("--n_layers", type=int, default=8,
                        help="Number of mixer layers")
    parser.add_argument("--token_mix_dim", type=int, default=160,
                        help="Token mixing MLP hidden dim")
    parser.add_argument("--channel_mix_dim", type=int, default=1280,
                        help="Channel mixing MLP hidden dim")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Training
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size (number of random walks per step)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="LR warmup steps")
    parser.add_argument("--total_steps", type=int, default=500_000,
                        help="Max training steps")

    # Curriculum
    parser.add_argument("--curriculum_start", type=int, default=3,
                        help="Initial max walk depth")
    parser.add_argument("--curriculum_end", type=int, default=40,
                        help="Final max walk depth (5x5 needs more moves)")
    parser.add_argument("--curriculum_ramp_steps", type=int, default=100_000,
                        help="Steps over which to ramp depth")

    # Bootstrap
    parser.add_argument("--bootstrap_after", type=int, default=30_000,
                        help="Switch to bootstrap targets after this many steps")

    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=5_000,
                        help="Steps between evaluations")
    parser.add_argument("--eval_num_cubes", type=int, default=20,
                        help="Cubes per eval (5x5 beam search is slower)")
    parser.add_argument("--eval_scramble_length", type=int, default=30,
                        help="Scramble length for eval")
    parser.add_argument("--eval_max_steps", type=int, default=200,
                        help="Max beam search steps in eval")
    parser.add_argument("--eval_beam_widths", type=int, nargs="+", default=[16, 64],
                        help="Beam widths for evaluation")
    parser.add_argument("--eval_seed", type=int, default=42,
                        help="Seed for eval scrambles")

    # Logging and saving
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Steps between log prints")
    parser.add_argument("--save_interval", type=int, default=10_000,
                        help="Steps between checkpoints")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoints/value_5x5_centers",
                        help="Checkpoint directory")

    # System
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        eval_only(args)
    else:
        train(args)
