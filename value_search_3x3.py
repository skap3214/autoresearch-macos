"""
DeepCubeA-style value network + beam search for 3x3 Rubik's cube.

Trains a distance-prediction network via self-supervised random walks from solved,
then uses it as a heuristic in beam search to solve scrambled cubes.

Usage:
    python value_search_3x3.py [--hours 8] [--device cuda]

Based on: Agostinelli et al., "Solving the Rubik's Cube with Deep Reinforcement
Learning and Search" (Nature Machine Intelligence, 2019)
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rubiks import Cube, Move, FACE_ORDER, FACE_COLORS, random_scramble

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COLOR_TO_IDX = {c: i for i, c in enumerate(("W", "Y", "G", "B", "R", "O"))}
_FACE_TO_IDX = {f: i for i, f in enumerate(FACE_ORDER)}  # U R F D L B

# All 18 moves for 3x3: 6 faces x 3 turns
ALL_MOVES_3x3: list[Move] = []
for _face in FACE_ORDER:
    for _turns in (1, -1, 2):
        ALL_MOVES_3x3.append(Move(face=_face, depth=1, width=1, turns=_turns))

# Opposite faces
_OPPOSITE = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}

# Precompute inverse map for no-inverse pruning
_INVERSE_MOVE_IDX: dict[int, int] = {}
for _i, _m in enumerate(ALL_MOVES_3x3):
    _inv = _m.inverse()
    for _j, _m2 in enumerate(ALL_MOVES_3x3):
        if _m2.face == _inv.face and _m2.turns == _inv.turns:
            _INVERSE_MOVE_IDX[_i] = _j
            break


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def encode_cube_3x3(cube: Cube) -> torch.LongTensor:
    """Encode a 3x3 cube state as a flat tensor of 54 color indices (0-5)."""
    indices = []
    for face in FACE_ORDER:
        for row in cube.face_grid(face):
            for color in row:
                indices.append(_COLOR_TO_IDX[color])
    return torch.tensor(indices, dtype=torch.long)


def encode_cube_batch(cubes: list[Cube]) -> torch.LongTensor:
    """Encode a batch of cubes. Returns (B, 54) tensor."""
    batch = []
    for cube in cubes:
        indices = []
        for face in FACE_ORDER:
            for row in cube.face_grid(face):
                for color in row:
                    indices.append(_COLOR_TO_IDX[color])
        batch.append(indices)
    return torch.tensor(batch, dtype=torch.long)


# ---------------------------------------------------------------------------
# Move pruning
# ---------------------------------------------------------------------------

def get_valid_move_indices(last_move_idx: Optional[int]) -> list[int]:
    """Return indices of valid moves given the last move index.
    Prunes: don't undo last move, don't do same face after opposite face."""
    if last_move_idx is None:
        return list(range(18))
    inv_idx = _INVERSE_MOVE_IDX[last_move_idx]
    last_face = ALL_MOVES_3x3[last_move_idx].face
    valid = []
    for i in range(18):
        if i == inv_idx:
            continue
        # Also skip same-face redundancy (face already covered by a different turn)
        # Just skip the inverse for now
        valid.append(i)
    return valid


# ---------------------------------------------------------------------------
# Model: Per-sticker token encoder with MLP-Mixer blocks
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
        # Token mixing: transpose to (B, D, S), mix, transpose back
        y = self.norm1(x)
        y = y.transpose(1, 2)  # (B, D, S)
        y = self.token_mix(y)  # (B, D, S)
        y = y.transpose(1, 2)  # (B, S, D)
        x = x + y
        # Channel mixing
        y = self.norm2(x)
        y = self.channel_mix(y)  # (B, S, D)
        x = x + y
        return x


class DistanceNet3x3(nn.Module):
    """Per-sticker token encoder + MLP-Mixer for distance prediction.

    Each of 54 stickers gets:
      - color embedding (6 colors)
      - face embedding (6 faces)
      - position features (row, col normalized to [-1, 1])

    Then 6 MLP-Mixer layers, global average pool, scalar output.
    Target: ~10-20M parameters.
    """

    def __init__(
        self,
        n_stickers: int = 54,
        n_colors: int = 6,
        n_faces: int = 6,
        color_embed_dim: int = 64,
        face_embed_dim: int = 32,
        pos_dim: int = 4,  # row, col (2D) -> small MLP -> pos_dim
        d_model: int = 256,
        n_layers: int = 6,
        token_mix_dim: int = 128,
        channel_mix_dim: int = 512,
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

        # Distance head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Precompute face indices and position features for 54 stickers
        face_indices = []
        pos_features = []
        for face_idx, face in enumerate(FACE_ORDER):
            for r in range(3):
                for c in range(3):
                    face_indices.append(face_idx)
                    # Normalize row, col to [-1, 1]
                    pos_features.append([r / 1.0 - 1.0, c / 1.0 - 1.0])
        self.register_buffer("face_indices", torch.tensor(face_indices, dtype=torch.long))
        self.register_buffer("pos_features", torch.tensor(pos_features, dtype=torch.float32))

    def forward(self, stickers: torch.Tensor) -> torch.Tensor:
        """stickers: (B, 54) integer tensor of color indices 0-5.
        Returns: (B,) predicted distances."""
        B = stickers.size(0)

        # Color embedding: (B, 54, color_embed_dim)
        color_emb = self.color_embed(stickers)

        # Face embedding: (B, 54, face_embed_dim) - same for all batch items
        face_emb = self.face_embed(self.face_indices).unsqueeze(0).expand(B, -1, -1)

        # Position features: (B, 54, pos_dim)
        pos_emb = self.pos_proj(self.pos_features).unsqueeze(0).expand(B, -1, -1)

        # Concatenate and project
        x = torch.cat([color_emb, face_emb, pos_emb], dim=-1)  # (B, 54, input_dim)
        x = self.input_proj(x)  # (B, 54, d_model)

        # Mixer layers
        for layer in self.mixer_layers:
            x = layer(x)
        x = self.final_norm(x)

        # Global average pooling
        x = x.mean(dim=1)  # (B, d_model)

        # Distance prediction
        return self.head(x).squeeze(-1)  # (B,)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Self-supervised data generation: random walks from solved
# ---------------------------------------------------------------------------

def sample_walk_batch_3x3(
    batch_size: int,
    max_depth: int,
    rng: random.Random,
    include_solved: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate training data: random walks from solved state.

    For each walk: start from solved, apply d random non-backtracking moves
    (d uniform in 1..max_depth). Record every prefix state with its distance.

    Also optionally include the solved state with distance 0.

    Returns: (states, distances) tensors.
    """
    all_indices = []
    all_dists = []

    if include_solved:
        cube = Cube(3)
        idx = []
        for face in FACE_ORDER:
            for row in cube.face_grid(face):
                for color in row:
                    idx.append(_COLOR_TO_IDX[color])
        all_indices.append(idx)
        all_dists.append(0.0)

    for _ in range(batch_size):
        d = rng.randint(1, max_depth)
        cube = Cube(3)
        last_move_idx = None

        for step in range(d):
            valid = get_valid_move_indices(last_move_idx)
            move_idx = rng.choice(valid)
            move = ALL_MOVES_3x3[move_idx]
            cube.apply_move(move)
            last_move_idx = move_idx

            # Record this prefix state
            idx = []
            for face in FACE_ORDER:
                for row in cube.face_grid(face):
                    for color in row:
                        idx.append(_COLOR_TO_IDX[color])
            all_indices.append(idx)
            all_dists.append(float(step + 1))

    states = torch.tensor(all_indices, dtype=torch.long)
    distances = torch.tensor(all_dists, dtype=torch.float32)
    return states, distances


def sample_walk_endpoints_3x3(
    batch_size: int,
    max_depth: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate training data: only the endpoint of each random walk.
    More memory efficient -- one sample per walk instead of all prefixes.

    Returns: (states, distances) tensors of shape (batch_size, 54) and (batch_size,).
    """
    all_indices = []
    all_dists = []

    for _ in range(batch_size):
        d = rng.randint(1, max_depth)
        cube = Cube(3)
        last_move_idx = None

        for step in range(d):
            valid = get_valid_move_indices(last_move_idx)
            move_idx = rng.choice(valid)
            move = ALL_MOVES_3x3[move_idx]
            cube.apply_move(move)
            last_move_idx = move_idx

        idx = []
        for face in FACE_ORDER:
            for row in cube.face_grid(face):
                for color in row:
                    idx.append(_COLOR_TO_IDX[color])
        all_indices.append(idx)
        all_dists.append(float(d))

    states = torch.tensor(all_indices, dtype=torch.long)
    distances = torch.tensor(all_dists, dtype=torch.float32)
    return states, distances


# ---------------------------------------------------------------------------
# DeepCubeA-style data generation: one-step bootstrapping from value network
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_dcba_targets(
    model: DistanceNet3x3,
    batch_size: int,
    max_depth: int,
    rng: random.Random,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepCubeA-style target generation.

    1. Generate random scrambled states (walk endpoints)
    2. For each state, expand all 18 children
    3. Target = 1 + min(model(children))  (cost-to-go bootstrap)
    4. Solved states get target = 0

    Returns: (states, targets) where states is (B, 54) and targets is (B,).
    """
    model.eval()

    # Generate scrambled states
    cubes = []
    for _ in range(batch_size):
        d = rng.randint(1, max_depth)
        cube = Cube(3)
        last_move_idx = None
        for step in range(d):
            valid = get_valid_move_indices(last_move_idx)
            move_idx = rng.choice(valid)
            cube.apply_move(ALL_MOVES_3x3[move_idx])
            last_move_idx = move_idx
        cubes.append(cube)

    # Also include a few solved states
    n_solved = max(1, batch_size // 20)
    for _ in range(n_solved):
        cubes.append(Cube(3))

    all_states = encode_cube_batch(cubes).to(device)  # (B', 54)
    B = len(cubes)

    # Expand children for each state
    child_cubes = []
    for cube in cubes:
        for move in ALL_MOVES_3x3:
            child = cube.copy()
            child.apply_move(move)
            child_cubes.append(child)

    child_states = encode_cube_batch(child_cubes).to(device)  # (B'*18, 54)
    child_values = model(child_states)  # (B'*18,)
    child_values = child_values.view(B, 18)  # (B, 18)

    # Target = 1 + min over children
    targets = 1.0 + child_values.min(dim=1).values  # (B,)

    # Override: solved states get target = 0
    for i, cube in enumerate(cubes):
        if cube.is_solved():
            targets[i] = 0.0

    # Clamp targets to be non-negative
    targets = targets.clamp(min=0.0)

    return all_states, targets


# ---------------------------------------------------------------------------
# Beam search solver
# ---------------------------------------------------------------------------

@dataclass
class BeamItem:
    cube: Cube
    history: list[int]  # list of move indices
    last_move_idx: Optional[int]


@torch.no_grad()
def beam_search_3x3(
    model: DistanceNet3x3,
    cube: Cube,
    beam_width: int = 32,
    max_steps: int = 200,
    device: str = "cuda",
) -> tuple[bool, list[Move], int]:
    """Beam search using distance network as heuristic.

    Returns (solved, move_history, nodes_expanded).
    """
    model.eval()

    if cube.is_solved():
        return True, [], 0

    beam = [BeamItem(cube=cube.copy(), history=[], last_move_idx=None)]
    visited: set[str] = {cube.to_kociemba_string()}
    nodes_expanded = 0

    for step in range(max_steps):
        # Expand all beam items
        candidates: list[tuple[Cube, list[int], int, str]] = []
        for item in beam:
            if item.cube.is_solved():
                moves = [ALL_MOVES_3x3[i] for i in item.history]
                return True, moves, nodes_expanded

            valid_moves = get_valid_move_indices(item.last_move_idx)
            for move_idx in valid_moves:
                move = ALL_MOVES_3x3[move_idx]
                new_cube = item.cube.copy()
                new_cube.apply_move(move)
                state_key = new_cube.to_kociemba_string()
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
        states = encode_cube_batch([c for c, _, _, _ in candidates]).to(device)
        # Process in chunks to avoid OOM
        chunk_size = 4096
        all_scores = []
        for i in range(0, len(candidates), chunk_size):
            chunk = states[i:i + chunk_size]
            scores = model(chunk).cpu()
            all_scores.append(scores)
        pred_distances = torch.cat(all_scores, dim=0)

        # Sort by predicted distance (pure greedy heuristic -- no g-cost,
        # since we want shortest predicted distance, not shortest path so far)
        scored = []
        for i, (c, hist, last_idx, key) in enumerate(candidates):
            h = pred_distances[i].item()
            scored.append((h, i, c, hist, last_idx, key))

        scored.sort(key=lambda x: x[0])

        # Check for solved among candidates
        beam = []
        for h, i, c, hist, last_idx, key in scored:
            if c.is_solved():
                moves = [ALL_MOVES_3x3[mi] for mi in hist]
                return True, moves, nodes_expanded
            if len(beam) >= beam_width:
                continue  # still check for solved
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
    model: DistanceNet3x3,
    num_cubes: int = 100,
    scramble_length: int = 20,
    beam_width: int = 32,
    max_steps: int = 200,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """Evaluate beam search on random scrambles.

    Returns dict with solve_rate, mean_solution_length, mean_nodes.
    """
    rng = random.Random(seed)
    solved_count = 0
    total_solution_len = 0
    total_nodes = 0

    for i in range(num_cubes):
        scramble = random_scramble(3, scramble_length, rng, max_depth=1, max_width=1)
        cube = Cube(3)
        cube.apply_moves(scramble)

        solved, history, nodes = beam_search_3x3(
            model, cube, beam_width=beam_width, max_steps=max_steps, device=device
        )

        if solved:
            solved_count += 1
            total_solution_len += len(history)
        total_nodes += nodes

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
    model = DistanceNet3x3(
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
        # Cosine decay
        progress = (step - warmup_steps) / max(1, args.total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Wandb
    if use_wandb:
        wandb.init(
            project="rubiks-value-search-3x3",
            config=vars(args),
            name=f"dcba-3x3-{args.d_model}d-{args.n_layers}L",
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
    use_bootstrap = False  # Start with random walk supervision
    bootstrap_start_step = args.bootstrap_after

    print(f"Training for up to {args.hours} hours ({max_seconds}s)", flush=True)
    print(f"Phase 1: Random walk supervision (steps 0-{bootstrap_start_step})")
    print(f"Phase 2: DeepCubeA bootstrap (steps {bootstrap_start_step}+)")
    print(f"Curriculum: depth {args.curriculum_start} -> {args.curriculum_end}")
    print(flush=True)

    # Load checkpoint if exists
    latest_ckpt = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
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
            # DeepCubeA-style: bootstrap targets from value network
            states, targets = generate_dcba_targets(
                model, args.batch_size, curr_max_depth, rng, device
            )
            model.train()
        else:
            # Phase 1: supervised random walk distance
            states, targets = sample_walk_endpoints_3x3(
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
            for bw in args.eval_beam_widths:
                results = evaluate_solver(
                    model,
                    num_cubes=args.eval_num_cubes,
                    scramble_length=args.eval_scramble_length,
                    beam_width=bw,
                    max_steps=args.eval_max_steps,
                    device=device,
                    seed=args.eval_seed,
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
    for bw in [1, 8, 32, 128]:
        results = evaluate_solver(
            model,
            num_cubes=200,
            scramble_length=20,
            beam_width=bw,
            max_steps=300,
            device=device,
            seed=42,
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
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DeepCubeA-style value+search for 3x3 Rubik's cube")

    # Time budget
    parser.add_argument("--hours", type=float, default=8.0, help="Training time budget in hours")

    # Model architecture
    parser.add_argument("--d_model", type=int, default=512, help="Model hidden dimension")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of mixer layers")
    parser.add_argument("--token_mix_dim", type=int, default=256, help="Token mixing MLP hidden dim")
    parser.add_argument("--channel_mix_dim", type=int, default=2048, help="Channel mixing MLP hidden dim")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for data generation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="LR warmup steps")
    parser.add_argument("--total_steps", type=int, default=500_000, help="Max training steps")

    # Curriculum
    parser.add_argument("--curriculum_start", type=int, default=3, help="Initial max walk depth")
    parser.add_argument("--curriculum_end", type=int, default=26, help="Final max walk depth")
    parser.add_argument("--curriculum_ramp_steps", type=int, default=100_000,
                        help="Steps over which to ramp depth")

    # DeepCubeA bootstrap
    parser.add_argument("--bootstrap_after", type=int, default=20_000,
                        help="Switch to bootstrap targets after this many steps")

    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=5_000, help="Steps between evaluations")
    parser.add_argument("--eval_num_cubes", type=int, default=50, help="Cubes per eval")
    parser.add_argument("--eval_scramble_length", type=int, default=20, help="Scramble length for eval")
    parser.add_argument("--eval_max_steps", type=int, default=200, help="Max beam search steps")
    parser.add_argument("--eval_beam_widths", type=int, nargs="+", default=[1, 8, 32],
                        help="Beam widths for evaluation")
    parser.add_argument("--eval_seed", type=int, default=42, help="Seed for eval scrambles")

    # Logging and saving
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between log prints")
    parser.add_argument("--save_interval", type=int, default=10_000, help="Steps between checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/value_3x3",
                        help="Checkpoint directory")

    # System
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
