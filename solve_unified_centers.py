#!/usr/bin/env python3
"""
Unified NxN center solver using a transformer architecture.

Handles variable cube sizes (4x4 through 7x7) with a single model.
Uses per-center-sticker tokens with color, face, position, and size embeddings
fed into an encoder-only transformer with policy and value heads.

Usage:
  python solve_unified_centers.py --train --hours 4 --sizes 4
  python solve_unified_centers.py --train --hours 4 --sizes 4,5
  python solve_unified_centers.py --eval --sizes 4,5
  python solve_unified_centers.py --solve --size 5
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys
import time
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

# Center stickers per face for each cube size
# N=4: inner 2x2 = 4 per face, N=5: inner 3x3 = 9, N=6: inner 4x4 = 16, N=7: inner 5x5 = 25
CENTER_GRID_SIZE = {4: 2, 5: 3, 6: 4, 7: 5}
CENTER_STICKERS_PER_FACE = {n: CENTER_GRID_SIZE[n] ** 2 for n in CENTER_GRID_SIZE}
TOTAL_CENTER_STICKERS = {n: CENTER_STICKERS_PER_FACE[n] * 6 for n in CENTER_GRID_SIZE}
# {4: 24, 5: 54, 6: 96, 7: 150}

MAX_CENTER_STICKERS = 150  # 7x7

# Size index mapping: 4->0, 5->1, 6->2, 7->3
SIZE_TO_IDX = {4: 0, 5: 1, 6: 2, 7: 3}
NUM_SIZES = 4

# Build the 36-move action space: 18 regular (width=1) + 18 wide (width=2)
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

DATA_DIR = _PROJECT_ROOT / "data_unified_centers"
MODEL_FILE = DATA_DIR / "unified_center_model.pt"

# Per-size data file locations
SIZE_DATA_FILES = {
    4: _PROJECT_ROOT / "data_4x4_teacher" / "center_data.pkl",
    5: _PROJECT_ROOT / "data_5x5_teacher" / "center_data.pkl",
    6: _PROJECT_ROOT / "data_6x6_teacher" / "center_data.pkl",
    7: _PROJECT_ROOT / "data_7x7_teacher" / "center_data.pkl",
}

# ---------------------------------------------------------------------------
# Center sticker extraction (generic for any size)
# ---------------------------------------------------------------------------

def extract_center_stickers_generic(cube: Cube) -> list[int]:
    """Extract center sticker color indices from an NxN cube.
    Returns a flat list of color indices, ordered by face then row-major within center grid.
    """
    n = cube.size
    g = CENTER_GRID_SIZE[n]  # inner grid dimension
    indices = []
    for face in FACE_ORDER:
        grid = cube.face_grid(face)
        for r in range(1, 1 + g):
            for c in range(1, 1 + g):
                indices.append(COLOR_TO_IDX[grid[r][c]])
    return indices


def build_token_metadata(cube_size: int) -> list[tuple[int, float, float, int]]:
    """Build per-token metadata: (face_idx, norm_row, norm_col, size_idx).
    Returns one tuple per center sticker token for the given cube size.
    """
    g = CENTER_GRID_SIZE[cube_size]
    size_idx = SIZE_TO_IDX[cube_size]
    metadata = []
    for face in FACE_ORDER:
        face_idx = FACE_TO_IDX[face]
        for r in range(g):
            for c in range(g):
                # Normalize row/col to [0, 1]
                norm_r = r / max(g - 1, 1)
                norm_c = c / max(g - 1, 1)
                metadata.append((face_idx, norm_r, norm_c, size_idx))
    return metadata


# ---------------------------------------------------------------------------
# Centers-done check (generic)
# ---------------------------------------------------------------------------

def centers_done_generic(cube: Cube) -> bool:
    """Check if all center stickers on each face are uniform for any NxN cube."""
    n = cube.size
    if n == 4:
        return centers_done_444(cube)
    g = CENTER_GRID_SIZE[n]
    for face in FACE_ORDER:
        grid = cube.face_grid(face)
        center_color = grid[1][1]
        for r in range(1, 1 + g):
            for c in range(1, 1 + g):
                if grid[r][c] != center_color:
                    return False
    return True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UnifiedCenterDataset(Dataset):
    """Dataset that holds examples from multiple cube sizes.
    Each example stores: (color_indices, face_indices, norm_positions, size_idx, action, distance, seq_len)
    Padded to MAX_CENTER_STICKERS.
    """

    def __init__(self, examples: list[dict]):
        """
        examples: list of dicts with keys:
          'colors': list[int] of length seq_len
          'size': int (4,5,6,7)
          'action': int
          'distance': float
        """
        n = len(examples)
        self.colors = torch.zeros(n, MAX_CENTER_STICKERS, dtype=torch.long)
        self.face_ids = torch.zeros(n, MAX_CENTER_STICKERS, dtype=torch.long)
        self.positions = torch.zeros(n, MAX_CENTER_STICKERS, 2, dtype=torch.float32)
        self.size_ids = torch.zeros(n, dtype=torch.long)
        self.actions = torch.zeros(n, dtype=torch.long)
        self.distances = torch.zeros(n, dtype=torch.float32)
        self.seq_lens = torch.zeros(n, dtype=torch.long)

        # Precompute metadata per size
        metadata_cache = {}
        for size in CENTER_GRID_SIZE:
            metadata_cache[size] = build_token_metadata(size)

        for i, ex in enumerate(examples):
            colors = ex['colors']
            size = ex['size']
            seq_len = len(colors)
            meta = metadata_cache[size]

            self.colors[i, :seq_len] = torch.tensor(colors, dtype=torch.long)
            for j, (face_idx, nr, nc, _) in enumerate(meta):
                self.face_ids[i, j] = face_idx
                self.positions[i, j, 0] = nr
                self.positions[i, j, 1] = nc
            self.size_ids[i] = SIZE_TO_IDX[size]
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
            self.actions[idx],
            self.distances[idx],
            self.seq_lens[idx],
        )


class SizeBalancedSampler(Sampler):
    """Samples equal numbers from each cube size per epoch.
    Each epoch iterates through min_count * num_sizes examples.
    """

    def __init__(self, size_ids: torch.Tensor, seed: int = 42):
        self.size_ids = size_ids
        self.seed = seed
        self.epoch = 0

        # Group indices by size
        self.size_groups = {}
        for idx in range(len(size_ids)):
            sid = size_ids[idx].item()
            if sid not in self.size_groups:
                self.size_groups[sid] = []
            self.size_groups[sid].append(idx)

        self.num_sizes = len(self.size_groups)
        # Use the size of the largest group, with oversampling for smaller groups
        self.max_count = max(len(v) for v in self.size_groups.values())

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = []
        for sid, group in self.size_groups.items():
            shuffled = group.copy()
            rng.shuffle(shuffled)
            # Oversample smaller groups to match largest
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

class UnifiedCenterTransformer(nn.Module):
    """Encoder-only transformer for unified NxN center solving.

    Per-center-sticker tokens with:
      - Color embedding (6 colors, dim=32)
      - Face embedding (6 faces, dim=16)
      - Position embedding: normalized (row, col) projected to dim=8
      - Size embedding (4 sizes, dim=8)
      Total per-token: 64 -> project to d_model

    Transformer encoder with mean pooling -> policy head + value head.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, n_layers: int = 8,
                 dropout: float = 0.1, n_actions: int = N_ACTIONS):
        super().__init__()
        self.d_model = d_model

        # Token feature embeddings
        self.color_embed = nn.Embedding(NUM_COLORS, 32)
        self.face_embed = nn.Embedding(NUM_FACES, 16)
        self.pos_proj = nn.Linear(2, 8)  # (norm_row, norm_col) -> 8
        self.size_embed = nn.Embedding(NUM_SIZES, 8)

        # Project concatenated features to d_model
        self.input_proj = nn.Linear(32 + 16 + 8 + 8, d_model)
        self.input_norm = nn.LayerNorm(d_model)

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
                seq_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            colors: (B, MAX_SEQ) LongTensor of color indices
            face_ids: (B, MAX_SEQ) LongTensor of face indices
            positions: (B, MAX_SEQ, 2) FloatTensor of normalized (row, col)
            size_ids: (B,) LongTensor of size indices
            seq_lens: (B,) LongTensor of actual sequence lengths
        Returns:
            policy_logits: (B, N_ACTIONS)
            value: (B,) predicted distance
        """
        B, S = colors.shape

        # Build per-token features
        color_emb = self.color_embed(colors)        # (B, S, 32)
        face_emb = self.face_embed(face_ids)        # (B, S, 16)
        pos_emb = self.pos_proj(positions)           # (B, S, 8)
        # Broadcast size embedding to all tokens
        size_emb = self.size_embed(size_ids)         # (B, 8)
        size_emb = size_emb.unsqueeze(1).expand(-1, S, -1)  # (B, S, 8)

        # Concatenate and project
        token_features = torch.cat([color_emb, face_emb, pos_emb, size_emb], dim=-1)  # (B, S, 64)
        x = self.input_proj(token_features)          # (B, S, d_model)
        x = self.input_norm(x)

        # Build padding mask: True where position >= seq_len (i.e., padded)
        pos_indices = torch.arange(S, device=colors.device).unsqueeze(0)  # (1, S)
        padding_mask = pos_indices >= seq_lens.unsqueeze(1)  # (B, S)

        # Transformer encoder
        x = self.encoder(x, src_key_padding_mask=padding_mask)  # (B, S, d_model)

        # Mean pooling over non-padded tokens
        mask_expanded = (~padding_mask).unsqueeze(-1).float()  # (B, S, 1)
        pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)  # (B, d_model)

        policy = self.policy_head(pooled)            # (B, N_ACTIONS)
        value = self.value_head(pooled).squeeze(-1)  # (B,)
        return policy, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_size_data(size: int) -> list[dict]:
    """Load training data for a given cube size from its pickle file.
    Returns list of dicts with keys: colors, size, action, distance.
    """
    data_file = SIZE_DATA_FILES[size]
    if not data_file.exists():
        print(f"WARNING: Data file {data_file} not found for size {size}, skipping.")
        return []

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    states = data["states"]
    actions = data["actions"]
    distances = data["distances"]
    num_cubes = data.get("num_cubes", "?")
    print(f"  Size {size}x{size}: {len(states)} examples from {num_cubes} cubes")

    examples = []
    for i in range(len(states)):
        examples.append({
            'colors': states[i],
            'size': size,
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
        print("ERROR: No training data found!")
        sys.exit(1)

    print(f"  Total: {len(all_examples)} examples across sizes {sizes}")

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(all_examples)
    split = int((1 - val_frac) * len(all_examples))
    train_examples = all_examples[:split]
    val_examples = all_examples[split:]

    print(f"  Train: {len(train_examples)}, Val: {len(val_examples)}")

    train_ds = UnifiedCenterDataset(train_examples)
    val_ds = UnifiedCenterDataset(val_examples)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def encode_state_for_inference(cube: Cube, device: torch.device):
    """Encode a single cube state for model inference. Returns tensors ready for forward()."""
    n = cube.size
    colors = extract_center_stickers_generic(cube)
    meta = build_token_metadata(n)
    seq_len = len(colors)

    colors_t = torch.zeros(1, MAX_CENTER_STICKERS, dtype=torch.long, device=device)
    face_ids_t = torch.zeros(1, MAX_CENTER_STICKERS, dtype=torch.long, device=device)
    positions_t = torch.zeros(1, MAX_CENTER_STICKERS, 2, dtype=torch.float32, device=device)
    size_ids_t = torch.tensor([SIZE_TO_IDX[n]], dtype=torch.long, device=device)
    seq_lens_t = torch.tensor([seq_len], dtype=torch.long, device=device)

    colors_t[0, :seq_len] = torch.tensor(colors, dtype=torch.long)
    for j, (face_idx, nr, nc, _) in enumerate(meta):
        face_ids_t[0, j] = face_idx
        positions_t[0, j, 0] = nr
        positions_t[0, j, 1] = nc

    return colors_t, face_ids_t, positions_t, size_ids_t, seq_lens_t


def encode_batch_for_inference(cubes: list[Cube], device: torch.device):
    """Encode a batch of cubes for model inference."""
    B = len(cubes)
    colors_t = torch.zeros(B, MAX_CENTER_STICKERS, dtype=torch.long, device=device)
    face_ids_t = torch.zeros(B, MAX_CENTER_STICKERS, dtype=torch.long, device=device)
    positions_t = torch.zeros(B, MAX_CENTER_STICKERS, 2, dtype=torch.float32, device=device)
    size_ids_t = torch.zeros(B, dtype=torch.long, device=device)
    seq_lens_t = torch.zeros(B, dtype=torch.long, device=device)

    # Cache metadata per size
    meta_cache = {}

    for i, cube in enumerate(cubes):
        n = cube.size
        if n not in meta_cache:
            meta_cache[n] = build_token_metadata(n)
        meta = meta_cache[n]
        colors = extract_center_stickers_generic(cube)
        seq_len = len(colors)

        colors_t[i, :seq_len] = torch.tensor(colors, dtype=torch.long)
        for j, (face_idx, nr, nc, _) in enumerate(meta):
            face_ids_t[i, j] = face_idx
            positions_t[i, j, 0] = nr
            positions_t[i, j, 1] = nc
        size_ids_t[i] = SIZE_TO_IDX[n]
        seq_lens_t[i] = seq_len

    return colors_t, face_ids_t, positions_t, size_ids_t, seq_lens_t


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(hours: float = 4.0, sizes: list[int] = None, batch_size: int = 256,
                lr: float = 3e-4, policy_weight: float = 1.0, value_weight: float = 0.1,
                resume: bool = False):
    """Train the unified center transformer on teacher data."""
    if sizes is None:
        sizes = [4]

    train_ds, val_ds = prepare_datasets(sizes)

    # Use size-balanced sampler for training
    train_sampler = SizeBalancedSampler(train_ds.size_ids, seed=42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnifiedCenterTransformer().to(device)

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
    total_steps_estimate = int(hours * 3600 / (batch_size / 256)) * 10  # rough
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

        # Per-size tracking
        size_correct = {}
        size_total = {}

        for batch in train_loader:
            colors, face_ids, positions, size_ids, actions, distances, seq_lens = batch
            colors = colors.to(device)
            face_ids = face_ids.to(device)
            positions = positions.to(device)
            size_ids = size_ids.to(device)
            actions = actions.to(device)
            distances = distances.to(device)
            seq_lens = seq_lens.to(device)

            policy_logits, value_pred = model(colors, face_ids, positions, size_ids, seq_lens)

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

            # Per-size accuracy tracking
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
                colors, face_ids, positions, size_ids, actions, distances, seq_lens = batch
                colors = colors.to(device)
                face_ids = face_ids.to(device)
                positions = positions.to(device)
                size_ids = size_ids.to(device)
                actions = actions.to(device)
                distances = distances.to(device)
                seq_lens = seq_lens.to(device)

                policy_logits, value_pred = model(colors, face_ids, positions, size_ids, seq_lens)
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
        size_strs = []
        for sid in sorted(val_size_total.keys()):
            actual_size = [k for k, v in SIZE_TO_IDX.items() if v == sid][0]
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
            }, MODEL_FILE)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete after {epoch - start_epoch} epochs. Best val_acc={best_val_acc:.4f}")


# ---------------------------------------------------------------------------
# Solving / Evaluation
# ---------------------------------------------------------------------------

def generate_scramble(rng: random.Random, cube_size: int, n_moves: int = None) -> list[Move]:
    """Generate a random scramble appropriate for the cube size."""
    if n_moves is None:
        n_moves = {4: 20, 5: 30, 6: 40, 7: 50}.get(cube_size, 30)
    moves = []
    for _ in range(n_moves):
        face = rng.choice(list(FACE_ORDER))
        turns = rng.choice([1, -1, 2])
        width = rng.choice([1, 2])
        moves.append(Move(face=face, depth=1, width=width, turns=turns))
    return moves


def solve_greedy(cube: Cube, model: UnifiedCenterTransformer, device: torch.device,
                 max_steps: int = None) -> tuple[bool, list[Move], int]:
    """Solve centers by greedily following the policy's top prediction."""
    n = cube.size
    if max_steps is None:
        max_steps = {4: 50, 5: 80, 6: 120, 7: 160}.get(n, 100)

    current = cube.copy()
    moves = []
    for step in range(max_steps):
        if centers_done_generic(current):
            return True, moves, step
        inp = encode_state_for_inference(current, device)
        with torch.no_grad():
            policy_logits, _ = model(*inp)
        action = policy_logits.argmax(dim=-1).item()
        move = ALL_MOVES[action]
        current.apply_move(move)
        moves.append(move)
    return centers_done_generic(current), moves, max_steps


def solve_beam(cube: Cube, model: UnifiedCenterTransformer, device: torch.device,
               beam_width: int = 32, max_steps: int = None, top_k: int = 5
               ) -> tuple[bool, list[Move], int]:
    """Solve centers using policy-guided beam search with value ranking."""
    n = cube.size
    if max_steps is None:
        max_steps = {4: 50, 5: 80, 6: 120, 7: 160}.get(n, 100)

    if centers_done_generic(cube):
        return True, [], 0

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
        inp = encode_batch_for_inference(beam_cubes, device)
        with torch.no_grad():
            policy_logits, values = model(*inp)
            probs = F.softmax(policy_logits, dim=-1)

        for i, item in enumerate(beam):
            top_actions = probs[i].topk(top_k).indices.tolist()
            for action in top_actions:
                move = ALL_MOVES[action]
                child = item.cube.copy()
                child.apply_move(move)
                nodes += 1

                if centers_done_generic(child):
                    return True, item.moves + [move], nodes

                key = child.to_kociemba_string()
                if key in visited:
                    continue
                visited.add(key)

                # Score child by value prediction
                child_inp = encode_state_for_inference(child, device)
                with torch.no_grad():
                    _, child_val = model(*child_inp)
                score = child_val.item()

                candidates.append(BeamItem(
                    cube=child,
                    moves=item.moves + [move],
                    score=score,
                ))

        candidates.sort(key=lambda c: c.score)
        beam = candidates[:beam_width]

    return False, [], nodes


def evaluate(sizes: list[int], num_cubes: int = 200, seed: int = 123,
             beam_width: int = 32, beam_cubes: int = 50):
    """Evaluate the trained model on random scrambles for each size."""
    if not MODEL_FILE.exists():
        print(f"ERROR: Model file {MODEL_FILE} not found. Run --train first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnifiedCenterTransformer().to(device)
    ckpt = torch.load(MODEL_FILE, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Trained on sizes: {ckpt.get('sizes_trained', '?')}")

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"  Evaluating {size}x{size}")
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

            if centers_done_generic(cube):
                greedy_solved += 1
                greedy_lengths.append(0)
                continue

            solved, moves, steps = solve_greedy(cube, model, device)
            if solved:
                greedy_solved += 1
                greedy_lengths.append(len(moves))

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

            if centers_done_generic(cube):
                beam_solved += 1
                beam_lengths.append(0)
                continue

            solved, moves, nodes = solve_beam(cube, model, device, beam_width=beam_width)
            if solved:
                beam_solved += 1
                beam_lengths.append(len(moves))

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{bn}] solved={beam_solved}")

        print(f"Beam solve rate: {beam_solved}/{bn} = {beam_solved/bn:.1%}")
        if beam_lengths:
            print(f"  Avg solution length: {sum(beam_lengths)/len(beam_lengths):.1f} moves")
            print(f"  Max solution length: {max(beam_lengths)} moves")


def solve_one(size: int, beam_width: int = 32, seed: int = None):
    """Scramble and solve a single cube, showing the solution."""
    if not MODEL_FILE.exists():
        print(f"ERROR: Model file {MODEL_FILE} not found. Run --train first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnifiedCenterTransformer().to(device)
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
    print(f"Centers done: {centers_done_generic(cube)}")

    # Try greedy first
    print("\nAttempting greedy solve...")
    solved, moves, steps = solve_greedy(cube, model, device)
    if solved:
        print(f"Greedy solved in {len(moves)} moves!")
        print(f"Solution: {' '.join(str(m) for m in moves)}")
        return

    # Try beam search
    print(f"\nGreedy failed, attempting beam search (width={beam_width})...")
    solved, moves, nodes = solve_beam(cube, model, device, beam_width=beam_width)
    if solved:
        print(f"Beam search solved in {len(moves)} moves ({nodes} nodes explored)!")
        print(f"Solution: {' '.join(str(m) for m in moves)}")
    else:
        print(f"Failed to solve centers ({nodes} nodes explored).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_sizes(s: str) -> list[int]:
    """Parse comma-separated size list like '4,5' into [4, 5]."""
    return sorted(int(x.strip()) for x in s.split(","))


def main():
    parser = argparse.ArgumentParser(
        description="Unified NxN center solver via transformer + teacher distillation"
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--solve", action="store_true", help="Solve one random cube")
    parser.add_argument("--sizes", type=str, default="4",
                        help="Comma-separated cube sizes for training/eval (e.g. '4,5')")
    parser.add_argument("--size", type=int, default=4,
                        help="Single cube size for --solve mode")
    parser.add_argument("--hours", type=float, default=4.0, help="Training time in hours")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from saved checkpoint")
    parser.add_argument("--num-cubes", type=int, default=200, help="Number of cubes for eval")
    parser.add_argument("--beam-width", type=int, default=32, help="Beam width for eval/solve")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for --solve")
    args = parser.parse_args()

    if not any([args.train, args.eval, args.solve]):
        parser.print_help()
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.train:
        sizes = parse_sizes(args.sizes)
        print(f"Training unified center transformer on sizes: {sizes}")
        print(f"Training for {args.hours} hours, batch_size={args.batch_size}, lr={args.lr}")
        train_model(hours=args.hours, sizes=sizes, batch_size=args.batch_size,
                     lr=args.lr, resume=args.resume)

    if args.eval:
        sizes = parse_sizes(args.sizes)
        evaluate(sizes, num_cubes=args.num_cubes, beam_width=args.beam_width)

    if args.solve:
        solve_one(args.size, beam_width=args.beam_width, seed=args.seed)


if __name__ == "__main__":
    main()
