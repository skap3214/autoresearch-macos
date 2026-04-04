#!/usr/bin/env python3
"""
Fully neural NxN Rubik's cube solver. NO classical solver at inference time.

Uses full-sticker token input with a stage-conditioned encoder-only transformer.
Teacher data (dwalton solver) used for training only.

Supported sizes: 2x2, 3x3, 4x4, 5x5

Architecture:
  - Full-sticker tokens: color(32d) + face(16d) + position(8d) + size(8d) + type(8d) = 72 -> d_model
  - Learnable stage token prepended to sequence
  - Encoder-only transformer: d_model=384, n_heads=8, n_layers=10
  - Policy head (36-way with masking) + Value head (distance to next stage boundary)

Stages:
  - 2x2/3x3: SCRAMBLED -> SOLVING -> SOLVED
  - 4x4/5x5: SCRAMBLED -> REDUCTION -> FINISH -> SOLVED

Usage:
  python solve_neural_nxn.py --generate --sizes 3,4 --num-cubes 5000
  python solve_neural_nxn.py --train --sizes 3,4 --hours 4
  python solve_neural_nxn.py --eval --sizes 3,4
  python solve_neural_nxn.py --solve --size 4
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rubiks import Cube, Move, FACE_ORDER, centers_done_444

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLOR_TO_IDX = {c: i for i, c in enumerate(("W", "Y", "G", "B", "R", "O"))}
NUM_COLORS = 6
FACE_TO_IDX = {f: i for i, f in enumerate(FACE_ORDER)}
NUM_FACES = 6

# Sticker type: center, edge, corner
STICKER_TYPE_CENTER = 0
STICKER_TYPE_EDGE = 1
STICKER_TYPE_CORNER = 2
NUM_STICKER_TYPES = 3

# Size index mapping: 2->0, 3->1, 4->2, 5->3
SIZE_TO_IDX = {2: 0, 3: 1, 4: 2, 5: 3}
NUM_SIZES = 4

# Total stickers per cube: n^2 * 6
TOTAL_STICKERS = {2: 24, 3: 54, 4: 96, 5: 150}
MAX_STICKERS = 150  # 5x5

# Stage definitions
STAGE_SCRAMBLED = 0  # Not used in training data (just initial)
STAGE_REDUCTION = 1  # Centers+edges phase for 4x4/5x5
STAGE_SOLVING = 2    # 3x3-equivalent solve / direct solve for 2x2/3x3
STAGE_SOLVED = 3     # Terminal
NUM_STAGES = 4

STAGE_NAMES = {0: "SCRAMBLED", 1: "REDUCTION", 2: "SOLVING", 3: "SOLVED"}

# Build 36-move action space: 18 regular (width=1) + 18 wide (width=2)
ALL_MOVES: list[Move] = []
for _face in FACE_ORDER:
    for _turns in (1, -1, 2):
        ALL_MOVES.append(Move(face=_face, depth=1, width=1, turns=_turns))
for _face in FACE_ORDER:
    for _turns in (1, -1, 2):
        ALL_MOVES.append(Move(face=_face, depth=1, width=2, turns=_turns))

N_ACTIONS = len(ALL_MOVES)  # 36

MOVE_TO_IDX = {}
for _i, _m in enumerate(ALL_MOVES):
    MOVE_TO_IDX[(_m.face, _m.width, _m.turns)] = _i

# For 2x2/3x3, only width=1 moves are valid (indices 0-17)
SMALL_CUBE_VALID_ACTIONS = list(range(18))

DATA_DIR = _PROJECT_ROOT / "data_neural_nxn"
MODEL_FILE = DATA_DIR / "neural_nxn_model.pt"

# ---------------------------------------------------------------------------
# Sticker extraction and metadata
# ---------------------------------------------------------------------------

def classify_sticker(n: int, row: int, col: int) -> int:
    """Classify a sticker position as center, edge, or corner.
    row, col are 0-indexed within the face grid.
    """
    is_edge_row = (row == 0 or row == n - 1)
    is_edge_col = (col == 0 or col == n - 1)
    if is_edge_row and is_edge_col:
        return STICKER_TYPE_CORNER
    if is_edge_row or is_edge_col:
        return STICKER_TYPE_EDGE
    return STICKER_TYPE_CENTER


def extract_all_stickers(cube: Cube) -> list[int]:
    """Extract ALL sticker color indices from an NxN cube.
    Returns a flat list of color indices, ordered by face (URFDLB) then row-major.
    """
    n = cube.size
    indices = []
    for face in FACE_ORDER:
        grid = cube.face_grid(face)
        for r in range(n):
            for c in range(n):
                indices.append(COLOR_TO_IDX[grid[r][c]])
    return indices


def build_sticker_metadata(cube_size: int) -> list[tuple[int, float, float, int, int]]:
    """Build per-sticker metadata: (face_idx, norm_row, norm_col, size_idx, sticker_type).
    Returns one tuple per sticker for the given cube size.
    """
    n = cube_size
    size_idx = SIZE_TO_IDX[n]
    metadata = []
    for face in FACE_ORDER:
        face_idx = FACE_TO_IDX[face]
        for r in range(n):
            for c in range(n):
                norm_r = r / max(n - 1, 1)
                norm_c = c / max(n - 1, 1)
                stype = classify_sticker(n, r, c)
                metadata.append((face_idx, norm_r, norm_c, size_idx, stype))
    return metadata


# ---------------------------------------------------------------------------
# Stage detection
# ---------------------------------------------------------------------------

def centers_done_555(cube: Cube) -> bool:
    """Check if all 9 center stickers on each face are uniform for 5x5."""
    if cube.size != 5:
        raise ValueError(f"centers_done_555 only valid for 5x5, got {cube.size}")
    for face in FACE_ORDER:
        grid = cube.face_grid(face)
        center_color = grid[1][1]
        for r in range(1, 4):
            for c in range(1, 4):
                if grid[r][c] != center_color:
                    return False
    return True


def centers_done_generic(cube: Cube) -> bool:
    """Check if centers are done for any NxN cube (N >= 4)."""
    n = cube.size
    if n <= 3:
        return True  # No separate center phase for 2x2/3x3
    if n == 4:
        return centers_done_444(cube)
    if n == 5:
        return centers_done_555(cube)
    raise ValueError(f"Unsupported size {n}")


def detect_stage(cube: Cube) -> int:
    """Detect the current stage of the cube.
    For 2x2/3x3: SOLVED or SOLVING
    For 4x4/5x5: SOLVED, SOLVING (reduced), or REDUCTION
    """
    n = cube.size
    if n <= 3:
        if (n == 2 and cube.has_uniform_faces()) or (n == 3 and cube.is_solved()):
            return STAGE_SOLVED
        return STAGE_SOLVING
    # 4x4 or 5x5
    if cube.is_solved():
        return STAGE_SOLVED
    if centers_done_generic(cube):
        # Centers done -> in the "finishing" phase (3x3 equivalent solve)
        return STAGE_SOLVING
    return STAGE_REDUCTION


# ---------------------------------------------------------------------------
# Move action helpers
# ---------------------------------------------------------------------------

def move_to_action_idx(move: Move) -> int:
    """Convert a Move to an action index (0-35)."""
    key = (move.face, move.width, move.turns)
    if key not in MOVE_TO_IDX:
        raise ValueError(f"Move {move} not in action space: {key}")
    return MOVE_TO_IDX[key]


def valid_action_mask(cube_size: int) -> list[int]:
    """Return list of valid action indices for the given cube size."""
    if cube_size <= 3:
        return SMALL_CUBE_VALID_ACTIONS
    return list(range(N_ACTIONS))


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_scramble(rng: random.Random, cube_size: int, n_moves: int = None) -> list[Move]:
    """Generate a random scramble appropriate for the cube size."""
    if n_moves is None:
        n_moves = {2: 12, 3: 20, 4: 30, 5: 40}.get(cube_size, 20)
    moves = []
    for _ in range(n_moves):
        face = rng.choice(list(FACE_ORDER))
        turns = rng.choice([1, -1, 2])
        if cube_size <= 3:
            width = 1
        else:
            width = rng.choice([1, 2])
        moves.append(Move(face=face, depth=1, width=width, turns=turns))
    return moves


def _get_solver(cube_size: int):
    """Import and return the appropriate teacher solver function."""
    from teacher_dwalton import (
        _ensure_solver_importable,
        solve_cube_222,
        solve_cube_333,
        solve_cube_444,
        solve_cube_555,
    )
    _ensure_solver_importable()
    return {2: solve_cube_222, 3: solve_cube_333, 4: solve_cube_444, 5: solve_cube_555}[cube_size]


def generate_data_for_size(cube_size: int, num_cubes: int, seed: int = 42) -> dict:
    """Generate training data for a single cube size.

    For each scrambled cube:
      1. Solve with dwalton teacher
      2. Replay solution, detect stage transitions
      3. Record (all_stickers, stage, action, distance_to_next_stage) for each step
    """
    solver_fn = _get_solver(cube_size)
    rng = random.Random(seed)

    all_stickers = []    # list of flat color index lists
    all_stages = []      # list of stage ints
    all_actions = []     # list of action ints (0-35)
    all_distances = []   # list of ints (distance to next stage boundary)

    successes = 0
    failures = 0
    already_solved = 0
    t0 = time.time()

    for i in range(num_cubes):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{num_cubes}] size={cube_size} successes={successes} "
                  f"failures={failures} ({rate:.1f} cubes/s)")

        cube = Cube(cube_size)
        scramble = generate_scramble(rng, cube_size)
        cube.apply_moves(scramble)

        # Check if already solved
        is_done = (cube_size == 2 and cube.has_uniform_faces()) or \
                  (cube_size >= 3 and cube.is_solved())
        if is_done:
            already_solved += 1
            continue

        # Solve with teacher
        solution = None
        for attempt in range(3):
            try:
                solution = solver_fn(cube.copy())
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  WARNING: Solver failed on cube {i} (size={cube_size}): {e}")
                continue

        if solution is None or len(solution) == 0:
            failures += 1
            continue

        # First pass: detect stages at each step
        sim = cube.copy()
        stages_at_step = []
        for move in solution:
            stages_at_step.append(detect_stage(sim))
            sim.apply_move(move)
        stages_at_step.append(detect_stage(sim))  # final state after last move

        # Find next stage boundary for each step
        n = len(solution)
        next_boundary = [n] * n
        current_boundary = n
        for idx in range(n - 1, -1, -1):
            if idx + 1 < len(stages_at_step) and stages_at_step[idx + 1] != stages_at_step[idx]:
                current_boundary = idx + 1
            next_boundary[idx] = current_boundary

        # Second pass: build training examples
        state_cube = cube.copy()
        for step_idx in range(n):
            move = solution[step_idx]
            stage = stages_at_step[step_idx]
            dist_to_next_stage = next_boundary[step_idx] - step_idx

            stickers = extract_all_stickers(state_cube)
            action = move_to_action_idx(move)

            all_stickers.append(stickers)
            all_stages.append(stage)
            all_actions.append(action)
            all_distances.append(dist_to_next_stage)

            state_cube.apply_move(move)

        successes += 1

    elapsed = time.time() - t0
    print(f"  Size {cube_size}: {successes} cubes solved, {failures} failures, "
          f"{already_solved} already-solved, {len(all_stickers)} examples, {elapsed:.1f}s")

    return {
        "stickers": all_stickers,
        "stages": all_stages,
        "actions": all_actions,
        "distances": all_distances,
        "cube_size": cube_size,
        "num_cubes": successes,
    }


def generate_data(sizes: list[int], num_cubes: int, seed: int = 42):
    """Generate training data for all specified sizes."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for size in sizes:
        print(f"\nGenerating data for {size}x{size} ({num_cubes} cubes)...")
        data = generate_data_for_size(size, num_cubes, seed=seed + size)
        data_file = DATA_DIR / f"data_{size}x{size}.pkl"
        with open(data_file, "wb") as f:
            pickle.dump(data, f)
        print(f"  Saved to {data_file}")
        print(f"  Total examples: {len(data['stickers'])}")
        if data['distances']:
            avg_dist = sum(data['distances']) / len(data['distances'])
            max_dist = max(data['distances'])
            print(f"  Avg distance to next stage: {avg_dist:.1f}, Max: {max_dist}")
        # Stage distribution
        from collections import Counter
        stage_counts = Counter(data['stages'])
        for s in sorted(stage_counts.keys()):
            print(f"    Stage {STAGE_NAMES[s]}: {stage_counts[s]} examples")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NeuralNxNDataset(Dataset):
    """Dataset for full-sticker stage-conditioned training.
    Each example: (colors, face_ids, positions, size_id, sticker_types, stage, action, distance, seq_len)
    Padded to MAX_STICKERS.
    """

    def __init__(self, examples: list[dict]):
        n = len(examples)
        self.colors = torch.zeros(n, MAX_STICKERS, dtype=torch.long)
        self.face_ids = torch.zeros(n, MAX_STICKERS, dtype=torch.long)
        self.positions = torch.zeros(n, MAX_STICKERS, 2, dtype=torch.float32)
        self.size_ids = torch.zeros(n, dtype=torch.long)
        self.sticker_types = torch.zeros(n, MAX_STICKERS, dtype=torch.long)
        self.stages = torch.zeros(n, dtype=torch.long)
        self.actions = torch.zeros(n, dtype=torch.long)
        self.distances = torch.zeros(n, dtype=torch.float32)
        self.seq_lens = torch.zeros(n, dtype=torch.long)

        # Precompute metadata per size
        meta_cache = {}
        for size in SIZE_TO_IDX:
            meta_cache[size] = build_sticker_metadata(size)

        for i, ex in enumerate(examples):
            colors = ex['stickers']
            size = ex['size']
            seq_len = len(colors)
            meta = meta_cache[size]

            self.colors[i, :seq_len] = torch.tensor(colors, dtype=torch.long)
            for j, (face_idx, nr, nc, _, stype) in enumerate(meta):
                self.face_ids[i, j] = face_idx
                self.positions[i, j, 0] = nr
                self.positions[i, j, 1] = nc
                self.sticker_types[i, j] = stype
            self.size_ids[i] = SIZE_TO_IDX[size]
            self.stages[i] = ex['stage']
            self.actions[i] = ex['action']
            self.distances[i] = ex['distance']
            self.seq_lens[i] = seq_len

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            self.colors[idx],
            self.face_ids[idx],
            self.positions[idx],
            self.size_ids[idx],
            self.sticker_types[idx],
            self.stages[idx],
            self.actions[idx],
            self.distances[idx],
            self.seq_lens[idx],
        )


class SizeBalancedSampler(Sampler):
    """Samples equal numbers from each cube size per epoch."""

    def __init__(self, size_ids: torch.Tensor, seed: int = 42):
        self.size_ids = size_ids
        self.seed = seed
        self.epoch = 0

        self.size_groups = {}
        for idx in range(len(size_ids)):
            sid = size_ids[idx].item()
            if sid not in self.size_groups:
                self.size_groups[sid] = []
            self.size_groups[sid].append(idx)

        self.num_sizes = len(self.size_groups)
        self.max_count = max(len(v) for v in self.size_groups.values())

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = []
        for sid, group in self.size_groups.items():
            shuffled = group.copy()
            rng.shuffle(shuffled)
            if len(shuffled) < self.max_count:
                repeats = (self.max_count // len(shuffled)) + 1
                shuffled = (shuffled * repeats)[:self.max_count]
            indices.extend(shuffled[:self.max_count])
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.max_count * self.num_sizes


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NeuralNxNTransformer(nn.Module):
    """Stage-conditioned encoder-only transformer for full NxN solving.

    Per-sticker tokens with:
      - Color embedding (6 colors, dim=32)
      - Face embedding (6 faces, dim=16)
      - Position embedding: normalized (row, col) projected to dim=8
      - Size embedding (4 sizes, dim=8)
      - Sticker type embedding (center/edge/corner, dim=8)
      Total per-token: 72 -> project to d_model

    Stage token prepended to sequence.
    Transformer encoder with mean pooling -> policy head + value head.
    """

    def __init__(self, d_model: int = 384, n_heads: int = 8, n_layers: int = 10,
                 dropout: float = 0.1, n_actions: int = N_ACTIONS):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions

        # Token feature embeddings
        self.color_embed = nn.Embedding(NUM_COLORS, 32)
        self.face_embed = nn.Embedding(NUM_FACES, 16)
        self.pos_proj = nn.Linear(2, 8)   # (norm_row, norm_col) -> 8
        self.size_embed = nn.Embedding(NUM_SIZES, 8)
        self.type_embed = nn.Embedding(NUM_STICKER_TYPES, 8)

        # Project concatenated features to d_model: 32+16+8+8+8 = 72
        self.input_proj = nn.Linear(72, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Stage token: learnable embedding prepended to sequence
        self.stage_embed = nn.Embedding(NUM_STAGES, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_actions),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, colors: torch.Tensor, face_ids: torch.Tensor,
                positions: torch.Tensor, size_ids: torch.Tensor,
                sticker_types: torch.Tensor, stages: torch.Tensor,
                seq_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            colors: (B, MAX_STICKERS) LongTensor of color indices
            face_ids: (B, MAX_STICKERS) LongTensor of face indices
            positions: (B, MAX_STICKERS, 2) FloatTensor of normalized (row, col)
            size_ids: (B,) LongTensor of size indices
            sticker_types: (B, MAX_STICKERS) LongTensor of sticker types
            stages: (B,) LongTensor of stage indices
            seq_lens: (B,) LongTensor of actual sticker sequence lengths
        Returns:
            policy_logits: (B, N_ACTIONS)
            value: (B,) predicted distance to next stage
        """
        B, S = colors.shape

        # Build per-sticker features
        color_emb = self.color_embed(colors)           # (B, S, 32)
        face_emb = self.face_embed(face_ids)           # (B, S, 16)
        pos_emb = self.pos_proj(positions)              # (B, S, 8)
        size_emb = self.size_embed(size_ids)            # (B, 8)
        size_emb = size_emb.unsqueeze(1).expand(-1, S, -1)  # (B, S, 8)
        type_emb = self.type_embed(sticker_types)       # (B, S, 8)

        # Concatenate and project
        token_features = torch.cat([color_emb, face_emb, pos_emb, size_emb, type_emb], dim=-1)  # (B, S, 72)
        sticker_tokens = self.input_proj(token_features)  # (B, S, d_model)
        sticker_tokens = self.input_norm(sticker_tokens)

        # Stage token: (B, 1, d_model)
        stage_token = self.stage_embed(stages).unsqueeze(1)  # (B, 1, d_model)

        # Prepend stage token to sequence: (B, 1+S, d_model)
        x = torch.cat([stage_token, sticker_tokens], dim=1)  # (B, 1+S, d_model)

        # Build padding mask: stage token (pos 0) is never padded
        # Sticker positions 1..S: padded if (pos-1) >= seq_len
        pos_indices = torch.arange(1 + S, device=colors.device).unsqueeze(0)  # (1, 1+S)
        # Position 0 = stage token (always valid)
        # Position j (j>=1) = sticker j-1, valid if j-1 < seq_len
        padding_mask = pos_indices >= (seq_lens.unsqueeze(1) + 1)  # (B, 1+S)

        # Transformer encoder
        x = self.encoder(x, src_key_padding_mask=padding_mask)  # (B, 1+S, d_model)

        # Mean pooling over non-padded tokens (including stage token)
        mask_expanded = (~padding_mask).unsqueeze(-1).float()  # (B, 1+S, 1)
        pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)  # (B, d_model)

        policy = self.policy_head(pooled)              # (B, N_ACTIONS)
        value = self.value_head(pooled).squeeze(-1)    # (B,)
        return policy, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_size_data(size: int) -> list[dict]:
    """Load training data for a given cube size."""
    data_file = DATA_DIR / f"data_{size}x{size}.pkl"
    if not data_file.exists():
        print(f"WARNING: Data file {data_file} not found for size {size}, skipping.")
        return []

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    stickers = data["stickers"]
    stages = data["stages"]
    actions = data["actions"]
    distances = data["distances"]
    num_cubes = data.get("num_cubes", "?")
    print(f"  Size {size}x{size}: {len(stickers)} examples from {num_cubes} cubes")

    examples = []
    for i in range(len(stickers)):
        examples.append({
            'stickers': stickers[i],
            'size': size,
            'stage': stages[i],
            'action': actions[i],
            'distance': float(distances[i]),
        })
    return examples


def prepare_datasets(sizes: list[int], val_frac: float = 0.1, seed: int = 42):
    """Load data for specified sizes and split into train/val."""
    print("Loading data...")
    all_examples = []
    for size in sizes:
        examples = load_size_data(size)
        all_examples.extend(examples)

    if not all_examples:
        print("ERROR: No training data found! Run --generate first.")
        sys.exit(1)

    print(f"  Total: {len(all_examples)} examples across sizes {sizes}")

    rng = random.Random(seed)
    rng.shuffle(all_examples)
    split = int((1 - val_frac) * len(all_examples))
    train_examples = all_examples[:split]
    val_examples = all_examples[split:]

    print(f"  Train: {len(train_examples)}, Val: {len(val_examples)}")

    train_ds = NeuralNxNDataset(train_examples)
    val_ds = NeuralNxNDataset(val_examples)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

# Cache metadata per size
_META_CACHE = {}

def _get_meta(size: int):
    if size not in _META_CACHE:
        _META_CACHE[size] = build_sticker_metadata(size)
    return _META_CACHE[size]


def encode_state_for_inference(cube: Cube, stage: int, device: torch.device):
    """Encode a single cube state + stage for model inference."""
    n = cube.size
    colors = extract_all_stickers(cube)
    meta = _get_meta(n)
    seq_len = len(colors)

    colors_t = torch.zeros(1, MAX_STICKERS, dtype=torch.long, device=device)
    face_ids_t = torch.zeros(1, MAX_STICKERS, dtype=torch.long, device=device)
    positions_t = torch.zeros(1, MAX_STICKERS, 2, dtype=torch.float32, device=device)
    sticker_types_t = torch.zeros(1, MAX_STICKERS, dtype=torch.long, device=device)
    size_ids_t = torch.tensor([SIZE_TO_IDX[n]], dtype=torch.long, device=device)
    stages_t = torch.tensor([stage], dtype=torch.long, device=device)
    seq_lens_t = torch.tensor([seq_len], dtype=torch.long, device=device)

    colors_t[0, :seq_len] = torch.tensor(colors, dtype=torch.long)
    for j, (face_idx, nr, nc, _, stype) in enumerate(meta):
        face_ids_t[0, j] = face_idx
        positions_t[0, j, 0] = nr
        positions_t[0, j, 1] = nc
        sticker_types_t[0, j] = stype

    return colors_t, face_ids_t, positions_t, size_ids_t, sticker_types_t, stages_t, seq_lens_t


def encode_batch_for_inference(cubes: list[Cube], stages: list[int], device: torch.device):
    """Encode a batch of cubes + stages for model inference."""
    B = len(cubes)
    colors_t = torch.zeros(B, MAX_STICKERS, dtype=torch.long, device=device)
    face_ids_t = torch.zeros(B, MAX_STICKERS, dtype=torch.long, device=device)
    positions_t = torch.zeros(B, MAX_STICKERS, 2, dtype=torch.float32, device=device)
    sticker_types_t = torch.zeros(B, MAX_STICKERS, dtype=torch.long, device=device)
    size_ids_t = torch.zeros(B, dtype=torch.long, device=device)
    stages_t = torch.tensor(stages, dtype=torch.long, device=device)
    seq_lens_t = torch.zeros(B, dtype=torch.long, device=device)

    for i, cube in enumerate(cubes):
        n = cube.size
        meta = _get_meta(n)
        colors = extract_all_stickers(cube)
        seq_len = len(colors)

        colors_t[i, :seq_len] = torch.tensor(colors, dtype=torch.long)
        for j, (face_idx, nr, nc, _, stype) in enumerate(meta):
            face_ids_t[i, j] = face_idx
            positions_t[i, j, 0] = nr
            positions_t[i, j, 1] = nc
            sticker_types_t[i, j] = stype
        size_ids_t[i] = SIZE_TO_IDX[n]
        seq_lens_t[i] = seq_len

    return colors_t, face_ids_t, positions_t, size_ids_t, sticker_types_t, stages_t, seq_lens_t


def is_fully_solved(cube: Cube) -> bool:
    """Check if cube is fully solved (works for any size)."""
    if cube.size == 2:
        return cube.has_uniform_faces()
    return cube.is_solved()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(hours: float = 4.0, sizes: list[int] = None, batch_size: int = 256,
                lr: float = 3e-4, policy_weight: float = 1.0, value_weight: float = 0.1,
                resume: bool = False):
    """Train the neural NxN transformer."""
    if sizes is None:
        sizes = [3, 4]

    train_ds, val_ds = prepare_datasets(sizes)

    train_sampler = SizeBalancedSampler(train_ds.size_ids, seed=42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNxNTransformer().to(device)

    start_epoch = 0
    best_val_acc = 0.0

    if resume and MODEL_FILE.exists():
        ckpt = torch.load(MODEL_FILE, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_acc = ckpt.get("val_acc", 0.0)
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch} (val_acc={best_val_acc:.4f})")

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Device: {device}")
    print(f"Training on sizes: {sizes}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup + cosine schedule
    warmup_steps = 500
    total_steps_estimate = int(hours * 3600 / (batch_size / 256)) * 10
    step_count = 0

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps_estimate - warmup_steps, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    t0 = time.time()
    deadline = t0 + hours * 3600
    epoch = start_epoch

    while time.time() < deadline:
        epoch += 1
        train_sampler.set_epoch(epoch)
        model.train()

        total_loss = 0.0
        total_ploss = 0.0
        total_vloss = 0.0
        correct = 0
        total = 0

        size_correct = {}
        size_total = {}

        for batch in train_loader:
            colors, face_ids, positions, size_ids, sticker_types, stages, actions, distances, seq_lens = batch
            colors = colors.to(device)
            face_ids = face_ids.to(device)
            positions = positions.to(device)
            size_ids = size_ids.to(device)
            sticker_types = sticker_types.to(device)
            stages = stages.to(device)
            actions = actions.to(device)
            distances = distances.to(device)
            seq_lens = seq_lens.to(device)

            policy_logits, value_pred = model(
                colors, face_ids, positions, size_ids, sticker_types, stages, seq_lens
            )

            policy_loss = F.cross_entropy(policy_logits, actions)
            value_loss = F.smooth_l1_loss(value_pred, distances)
            loss = policy_weight * policy_loss + value_weight * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step_count += 1
            scheduler.step()

            bs = colors.size(0)
            total_loss += loss.item() * bs
            total_ploss += policy_loss.item() * bs
            total_vloss += value_loss.item() * bs
            preds = policy_logits.argmax(dim=-1)
            correct += (preds == actions).sum().item()
            total += bs

            for sid in size_ids.unique().tolist():
                mask = size_ids == sid
                sc = (preds[mask] == actions[mask]).sum().item()
                st = mask.sum().item()
                size_correct[sid] = size_correct.get(sid, 0) + sc
                size_total[sid] = size_total.get(sid, 0) + st

            if time.time() >= deadline:
                break

        if total == 0:
            continue

        avg_loss = total_loss / total
        avg_ploss = total_ploss / total
        avg_vloss = total_vloss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_vloss = 0.0
        val_size_correct = {}
        val_size_total = {}

        with torch.no_grad():
            for batch in val_loader:
                colors, face_ids, positions, size_ids, sticker_types, stages, actions, distances, seq_lens = batch
                colors = colors.to(device)
                face_ids = face_ids.to(device)
                positions = positions.to(device)
                size_ids = size_ids.to(device)
                sticker_types = sticker_types.to(device)
                stages = stages.to(device)
                actions = actions.to(device)
                distances = distances.to(device)
                seq_lens = seq_lens.to(device)

                policy_logits, value_pred = model(
                    colors, face_ids, positions, size_ids, sticker_types, stages, seq_lens
                )
                preds = policy_logits.argmax(dim=-1)
                val_correct += (preds == actions).sum().item()
                val_total += colors.size(0)
                val_vloss += F.smooth_l1_loss(value_pred, distances, reduction='sum').item()

                for sid in size_ids.unique().tolist():
                    mask = size_ids == sid
                    sc = (preds[mask] == actions[mask]).sum().item()
                    st = mask.sum().item()
                    val_size_correct[sid] = val_size_correct.get(sid, 0) + sc
                    val_size_total[sid] = val_size_total.get(sid, 0) + st

        val_acc = val_correct / val_total if val_total > 0 else 0
        val_vloss_avg = val_vloss / val_total if val_total > 0 else 0
        elapsed = time.time() - t0

        # Per-size val accuracy string
        idx_to_size = {v: k for k, v in SIZE_TO_IDX.items()}
        size_strs = []
        for sid in sorted(val_size_total.keys()):
            actual_size = idx_to_size[sid]
            acc = val_size_correct.get(sid, 0) / val_size_total[sid] if val_size_total.get(sid, 0) > 0 else 0
            size_strs.append(f"{actual_size}x{actual_size}={acc:.3f}")

        print(f"Epoch {epoch:3d} ({elapsed/60:.1f}m) | "
              f"loss={avg_loss:.4f} ploss={avg_ploss:.4f} vloss={avg_vloss:.4f} | "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} [{', '.join(size_strs)}] | "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_vloss": val_vloss_avg,
                "sizes_trained": sizes,
                "d_model": 384,
                "n_heads": 8,
                "n_layers": 10,
            }, MODEL_FILE)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete after {epoch - start_epoch} epochs. Best val_acc={best_val_acc:.4f}")


# ---------------------------------------------------------------------------
# Solving (fully neural, NO classical solver)
# ---------------------------------------------------------------------------

def solve_greedy(cube: Cube, model: NeuralNxNTransformer, device: torch.device,
                 max_steps: int = None) -> tuple[bool, list[Move], int]:
    """Solve cube by greedily following the policy's top prediction.
    Stage is detected at each step and fed to the model.
    """
    n = cube.size
    if max_steps is None:
        max_steps = {2: 30, 3: 50, 4: 100, 5: 150}.get(n, 100)

    valid_actions = valid_action_mask(n)
    current = cube.copy()
    moves = []

    for step in range(max_steps):
        if is_fully_solved(current):
            return True, moves, step

        stage = detect_stage(current)
        inp = encode_state_for_inference(current, stage, device)
        with torch.no_grad():
            policy_logits, _ = model(*inp)

        # Mask invalid actions for small cubes
        if n <= 3:
            mask = torch.full((N_ACTIONS,), float('-inf'), device=device)
            mask[valid_actions] = 0.0
            policy_logits = policy_logits + mask.unsqueeze(0)

        action = policy_logits.argmax(dim=-1).item()
        move = ALL_MOVES[action]
        current.apply_move(move)
        moves.append(move)

    return is_fully_solved(current), moves, max_steps


def solve_beam(cube: Cube, model: NeuralNxNTransformer, device: torch.device,
               beam_width: int = 32, max_steps: int = None, top_k: int = 8
               ) -> tuple[bool, list[Move], int]:
    """Solve cube using policy-guided beam search with value ranking.
    Stage is detected at each step. Fully neural -- no classical solver.
    """
    n = cube.size
    if max_steps is None:
        max_steps = {2: 30, 3: 50, 4: 80, 5: 120}.get(n, 80)

    if is_fully_solved(cube):
        return True, [], 0

    valid_actions = valid_action_mask(n)

    @dataclass
    class BeamItem:
        cube: Cube
        moves: list
        score: float

    beam = [BeamItem(cube=cube.copy(), moves=[], score=0.0)]
    visited = {cube.to_kociemba_string()}
    nodes = 0

    for step in range(max_steps):
        if not beam:
            break

        candidates = []

        # Batch inference for all beam items
        beam_cubes = [item.cube for item in beam]
        beam_stages = [detect_stage(item.cube) for item in beam]
        inp = encode_batch_for_inference(beam_cubes, beam_stages, device)
        with torch.no_grad():
            policy_logits, values = model(*inp)

        # Mask invalid actions for small cubes
        if n <= 3:
            mask = torch.full((N_ACTIONS,), float('-inf'), device=device)
            mask[valid_actions] = 0.0
            policy_logits = policy_logits + mask.unsqueeze(0)

        probs = F.softmax(policy_logits, dim=-1)

        for i, item in enumerate(beam):
            top_actions = probs[i].topk(min(top_k, len(valid_actions))).indices.tolist()
            for action in top_actions:
                move = ALL_MOVES[action]
                child = item.cube.copy()
                child.apply_move(move)
                nodes += 1

                if is_fully_solved(child):
                    return True, item.moves + [move], nodes

                key = child.to_kociemba_string()
                if key in visited:
                    continue
                visited.add(key)

                # Score child by value prediction
                child_stage = detect_stage(child)
                child_inp = encode_state_for_inference(child, child_stage, device)
                with torch.no_grad():
                    _, child_val = model(*child_inp)
                score = child_val.item()

                candidates.append(BeamItem(
                    cube=child,
                    moves=item.moves + [move],
                    score=score,
                ))

        # Keep top beam_width by lowest predicted distance
        candidates.sort(key=lambda c: c.score)
        beam = candidates[:beam_width]

    return False, [], nodes


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(sizes: list[int], num_cubes: int = 200, seed: int = 123,
             beam_width: int = 32, beam_cubes: int = 50):
    """Evaluate the fully neural solver on random scrambles."""
    if not MODEL_FILE.exists():
        print(f"ERROR: Model file {MODEL_FILE} not found. Run --train first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNxNTransformer().to(device)
    ckpt = torch.load(MODEL_FILE, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Trained on sizes: {ckpt.get('sizes_trained', '?')}")
    print(f"NOTE: Fully neural solver -- NO classical solver used at inference time")

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"  Evaluating {size}x{size} (fully neural)")
        print(f"{'='*60}")

        rng = random.Random(seed)

        # Greedy
        print(f"\n--- Greedy ({num_cubes} cubes) ---")
        greedy_solved = 0
        greedy_lengths = []
        for i in range(num_cubes):
            cube = Cube(size)
            scramble = generate_scramble(rng, size)
            cube.apply_moves(scramble)

            if is_fully_solved(cube):
                greedy_solved += 1
                greedy_lengths.append(0)
                continue

            solved, moves, steps = solve_greedy(cube, model, device)
            if solved:
                greedy_solved += 1
                greedy_lengths.append(len(moves))

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{num_cubes}] greedy_solved={greedy_solved}")

        print(f"Greedy solve rate: {greedy_solved}/{num_cubes} = {greedy_solved/num_cubes:.1%}")
        if greedy_lengths:
            print(f"  Avg solution length: {sum(greedy_lengths)/len(greedy_lengths):.1f} moves")
            print(f"  Max solution length: {max(greedy_lengths)} moves")

        # Beam search
        bn = min(num_cubes, beam_cubes)
        print(f"\n--- Beam Search ({bn} cubes, beam={beam_width}) ---")
        rng2 = random.Random(seed)
        beam_solved = 0
        beam_lengths = []
        for i in range(bn):
            cube = Cube(size)
            scramble = generate_scramble(rng2, size)
            cube.apply_moves(scramble)

            if is_fully_solved(cube):
                beam_solved += 1
                beam_lengths.append(0)
                continue

            solved, moves, nodes = solve_beam(cube, model, device, beam_width=beam_width)
            if solved:
                beam_solved += 1
                beam_lengths.append(len(moves))

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{bn}] beam_solved={beam_solved}")

        print(f"Beam solve rate: {beam_solved}/{bn} = {beam_solved/bn:.1%}")
        if beam_lengths:
            print(f"  Avg solution length: {sum(beam_lengths)/len(beam_lengths):.1f} moves")
            print(f"  Max solution length: {max(beam_lengths)} moves")


def solve_one(size: int, beam_width: int = 32, seed: int = None):
    """Scramble and solve a single cube, showing the solution. Fully neural."""
    if not MODEL_FILE.exists():
        print(f"ERROR: Model file {MODEL_FILE} not found. Run --train first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNxNTransformer().to(device)
    ckpt = torch.load(MODEL_FILE, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if seed is None:
        seed = int(time.time())
    rng = random.Random(seed)

    cube = Cube(size)
    scramble = generate_scramble(rng, size)
    cube.apply_moves(scramble)

    print(f"Scrambled {size}x{size} cube (seed={seed})")
    print(f"Fully solved: {is_fully_solved(cube)}")
    if size >= 4:
        print(f"Centers done: {centers_done_generic(cube)}")
    print(f"Current stage: {STAGE_NAMES[detect_stage(cube)]}")
    print(f"NOTE: Fully neural solver -- NO classical solver")

    # Try greedy first
    print("\nAttempting greedy solve...")
    t0 = time.time()
    solved, moves, steps = solve_greedy(cube, model, device)
    elapsed = time.time() - t0
    if solved:
        print(f"Greedy solved in {len(moves)} moves! ({elapsed:.2f}s)")
        for i, m in enumerate(moves):
            print(f"  {i+1}. {m}")
        return

    # Try beam search
    print(f"\nGreedy failed, attempting beam search (width={beam_width})...")
    t0 = time.time()
    solved, moves, nodes = solve_beam(cube, model, device, beam_width=beam_width)
    elapsed = time.time() - t0
    if solved:
        print(f"Beam search solved in {len(moves)} moves ({nodes} nodes, {elapsed:.2f}s)!")
        for i, m in enumerate(moves):
            print(f"  {i+1}. {m}")
    else:
        print(f"Failed to solve ({nodes} nodes explored, {elapsed:.2f}s).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_sizes(s: str) -> list[int]:
    """Parse comma-separated size list like '3,4' into [3, 4]."""
    sizes = sorted(int(x.strip()) for x in s.split(","))
    for s_val in sizes:
        if s_val not in SIZE_TO_IDX:
            raise ValueError(f"Unsupported size {s_val}. Supported: {list(SIZE_TO_IDX.keys())}")
    return sizes


def main():
    parser = argparse.ArgumentParser(
        description="Fully neural NxN Rubik's cube solver (NO classical solver at inference)"
    )
    parser.add_argument("--generate", action="store_true", help="Generate training data via teacher")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--solve", action="store_true", help="Solve one random cube")
    parser.add_argument("--sizes", type=str, default="3,4",
                        help="Comma-separated cube sizes (e.g. '2,3,4,5')")
    parser.add_argument("--size", type=int, default=3,
                        help="Single cube size for --solve mode")
    parser.add_argument("--num-cubes", type=int, default=5000,
                        help="Number of cubes for data generation")
    parser.add_argument("--hours", type=float, default=4.0, help="Training time in hours")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from saved checkpoint")
    parser.add_argument("--num-eval", type=int, default=200, help="Number of cubes for eval")
    parser.add_argument("--beam-width", type=int, default=32, help="Beam width for eval/solve")
    parser.add_argument("--beam-cubes", type=int, default=50, help="Number of cubes for beam eval")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for --solve")
    args = parser.parse_args()

    if not any([args.generate, args.train, args.eval, args.solve]):
        parser.print_help()
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.generate:
        sizes = parse_sizes(args.sizes)
        print(f"Generating training data for sizes: {sizes}")
        print(f"Using dwalton teacher solver (training data only)")
        generate_data(sizes, args.num_cubes, seed=args.seed or 42)

    if args.train:
        sizes = parse_sizes(args.sizes)
        print(f"Training neural NxN solver on sizes: {sizes}")
        print(f"Training for {args.hours} hours, batch_size={args.batch_size}, lr={args.lr}")
        train_model(hours=args.hours, sizes=sizes, batch_size=args.batch_size,
                     lr=args.lr, resume=args.resume)

    if args.eval:
        sizes = parse_sizes(args.sizes)
        evaluate(sizes, num_cubes=args.num_eval, beam_width=args.beam_width,
                 beam_cubes=args.beam_cubes)

    if args.solve:
        solve_one(args.size, beam_width=args.beam_width, seed=args.seed)


if __name__ == "__main__":
    main()
