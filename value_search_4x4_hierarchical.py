"""
Hierarchical reduction-based value+search solver for 4x4 Rubik's cubes.

Solves a 4x4 in three stages:
  Stage 1: Solve centers (all 6 center 2x2 blocks uniform color)
  Stage 2: Pair edges (all 12 edge pairs matched, preserving centers)
  Stage 3: Solve as 3x3 (hand off to trained 3x3 value+search model)

Each stage has its own MLP-Mixer heuristic network trained via self-supervised
random walks. Beam search guided by stage-specific heuristics chains the stages.

Usage:
    python value_search_4x4_hierarchical.py --stage 1 --hours 4   # train stage 1
    python value_search_4x4_hierarchical.py --stage 2 --hours 4   # train stage 2
    python value_search_4x4_hierarchical.py --solve --hours 0     # solve pipeline
    python value_search_4x4_hierarchical.py --eval --hours 0      # evaluate pipeline
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from rubiks import (
    Cube, Move, FACE_ORDER, FACE_COLORS, random_scramble,
    centers_done_444,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COLOR_TO_IDX = {c: i for i, c in enumerate(("W", "Y", "G", "B", "R", "O"))}
_FACE_TO_IDX = {f: i for i, f in enumerate(FACE_ORDER)}  # U R F D L B

# All 36 moves for 4x4: 6 faces x 3 turns x 2 widths (regular + wide)
ALL_MOVES_4x4: list[Move] = []
for _face in FACE_ORDER:
    for _turns in (1, -1, 2):
        ALL_MOVES_4x4.append(Move(face=_face, depth=1, width=1, turns=_turns))
for _face in FACE_ORDER:
    for _turns in (1, -1, 2):
        ALL_MOVES_4x4.append(Move(face=_face, depth=1, width=2, turns=_turns))

N_MOVES_4x4 = len(ALL_MOVES_4x4)  # 36

# 18 outer-layer-only moves (indices 0-17)
OUTER_MOVE_INDICES = list(range(18))
# 18 wide moves (indices 18-35)
WIDE_MOVE_INDICES = list(range(18, 36))

# Opposite faces
_OPPOSITE = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}

# Precompute inverse map
_INVERSE_MOVE_IDX_4x4: dict[int, int] = {}
for _i, _m in enumerate(ALL_MOVES_4x4):
    _inv = _m.inverse()
    for _j, _m2 in enumerate(ALL_MOVES_4x4):
        if (_m2.face == _inv.face and _m2.turns == _inv.turns
                and _m2.width == _inv.width and _m2.depth == _inv.depth):
            _INVERSE_MOVE_IDX_4x4[_i] = _j
            break

# 3x3 moves for stage 3
ALL_MOVES_3x3: list[Move] = []
for _face in FACE_ORDER:
    for _turns in (1, -1, 2):
        ALL_MOVES_3x3.append(Move(face=_face, depth=1, width=1, turns=_turns))


# ---------------------------------------------------------------------------
# 4x4 stage-specific metrics
# ---------------------------------------------------------------------------

def centers_solved_count_444(cube: Cube) -> int:
    """Count how many faces have their 4 center stickers uniform (0-6)."""
    count = 0
    for face in FACE_ORDER:
        grid = cube.face_grid(face)
        center_colors = {grid[1][1], grid[1][2], grid[2][1], grid[2][2]}
        if len(center_colors) == 1:
            count += 1
    return count


def paired_edge_count(cube: Cube) -> int:
    """Count how many of the 12 logical edges have their wing pair matched.

    A 4x4 has 24 wing cubies forming 12 edge pairs. Each edge_key (the
    position with the inner coordinate zeroed) identifies one logical edge
    slot with exactly 2 wing positions. The pair is 'paired' if the two
    wing positions sharing that edge_key have matching color sets.
    """
    limit = cube.limit  # 3 for 4x4

    # Group wing stickers by edge_key
    edge_groups: dict[tuple, dict[tuple, set]] = {}
    for (position, normal), color in cube.stickers.items():
        at_limit = sum(1 for c in position if abs(c) == limit)
        if at_limit != 2:
            continue
        edge_key = tuple(c if abs(c) == limit else 0 for c in position)
        if edge_key not in edge_groups:
            edge_groups[edge_key] = {}
        if position not in edge_groups[edge_key]:
            edge_groups[edge_key][position] = set()
        edge_groups[edge_key][position].add(color)

    paired = 0
    for ek, by_pos in edge_groups.items():
        positions = list(by_pos.values())
        if len(positions) == 2 and positions[0] == positions[1]:
            paired += 1
    return paired


def edges_paired(cube: Cube) -> bool:
    """Check if all 12 edge pairs are matched on a 4x4."""
    return paired_edge_count(cube) == 12


def stage1_heuristic_target(cube: Cube) -> float:
    """Distance-like target for stage 1: 0 when all centers solved."""
    return float(6 - centers_solved_count_444(cube))


def stage2_heuristic_target(cube: Cube) -> float:
    """Distance-like target for stage 2: 0 when all edges paired."""
    return float(12 - paired_edge_count(cube))


def stage1_is_done(cube: Cube) -> bool:
    """Check if stage 1 goal is met."""
    return centers_done_444(cube)


def stage2_is_done(cube: Cube) -> bool:
    """Check if stage 2 goal is met (all 12 edges paired).

    Centers are not checked here — stage 1 is responsible for solving centers,
    and beam search for stage 2 will naturally preserve them since the heuristic
    trains from reduced states that already have centers solved.
    """
    return edges_paired(cube)


# ---------------------------------------------------------------------------
# State encoding (4x4 = 96 stickers)
# ---------------------------------------------------------------------------

def encode_cube_4x4(cube: Cube) -> torch.LongTensor:
    """Encode a 4x4 cube state as a flat tensor of 96 color indices (0-5)."""
    indices = []
    for face in FACE_ORDER:
        for row in cube.face_grid(face):
            for color in row:
                indices.append(_COLOR_TO_IDX[color])
    return torch.tensor(indices, dtype=torch.long)


def encode_cube_batch_4x4(cubes: list[Cube]) -> torch.LongTensor:
    """Encode a batch of 4x4 cubes. Returns (B, 96) tensor."""
    batch = []
    for cube in cubes:
        indices = []
        for face in FACE_ORDER:
            for row in cube.face_grid(face):
                for color in row:
                    indices.append(_COLOR_TO_IDX[color])
        batch.append(indices)
    return torch.tensor(batch, dtype=torch.long)


def encode_cube_batch_3x3(cubes: list[Cube]) -> torch.LongTensor:
    """Encode a batch of 3x3 cubes. Returns (B, 54) tensor."""
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

def get_valid_move_indices_4x4(last_move_idx: Optional[int]) -> list[int]:
    """Return valid move indices (prune inverse of last move)."""
    if last_move_idx is None:
        return list(range(N_MOVES_4x4))
    inv_idx = _INVERSE_MOVE_IDX_4x4.get(last_move_idx)
    valid = []
    for i in range(N_MOVES_4x4):
        if inv_idx is not None and i == inv_idx:
            continue
        valid.append(i)
    return valid


def get_valid_move_indices_3x3(last_move_idx: Optional[int]) -> list[int]:
    """Return valid 3x3 move indices."""
    if last_move_idx is None:
        return list(range(18))
    # Find inverse
    m = ALL_MOVES_3x3[last_move_idx]
    inv = m.inverse()
    inv_idx = None
    for j, m2 in enumerate(ALL_MOVES_3x3):
        if m2.face == inv.face and m2.turns == inv.turns:
            inv_idx = j
            break
    valid = []
    for i in range(18):
        if inv_idx is not None and i == inv_idx:
            continue
        valid.append(i)
    return valid


# ---------------------------------------------------------------------------
# Model: Per-sticker MLP-Mixer (same architecture as value_search_3x3.py)
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
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mix(y)
        y = y.transpose(1, 2)
        x = x + y
        y = self.norm2(x)
        y = self.channel_mix(y)
        x = x + y
        return x


class StageDistanceNet(nn.Module):
    """Per-sticker token encoder + MLP-Mixer for stage-specific distance prediction.

    96 stickers (4x4 cube), each gets color+face+position embeddings.
    Smaller than full 4x4 solver (~5-10M params) since each stage is simpler.
    Output: scalar distance estimate for stage goal.
    """

    def __init__(
        self,
        n_stickers: int = 96,
        n_colors: int = 6,
        n_faces: int = 6,
        color_embed_dim: int = 48,
        face_embed_dim: int = 24,
        pos_dim: int = 4,
        d_model: int = 256,
        n_layers: int = 5,
        token_mix_dim: int = 128,
        channel_mix_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_stickers = n_stickers
        self.d_model = d_model

        self.color_embed = nn.Embedding(n_colors, color_embed_dim)
        self.face_embed = nn.Embedding(n_faces, face_embed_dim)
        self.pos_proj = nn.Linear(2, pos_dim)

        input_dim = color_embed_dim + face_embed_dim + pos_dim
        self.input_proj = nn.Linear(input_dim, d_model)

        self.mixer_layers = nn.ModuleList([
            MixerLayer(n_stickers, d_model, token_mix_dim, channel_mix_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Precompute face indices and positions for 96 stickers (4x4)
        face_indices = []
        pos_features = []
        grid_size = 4
        for face_idx, face in enumerate(FACE_ORDER):
            for r in range(grid_size):
                for c in range(grid_size):
                    face_indices.append(face_idx)
                    pos_features.append([r / 1.5 - 1.0, c / 1.5 - 1.0])
        self.register_buffer("face_indices", torch.tensor(face_indices, dtype=torch.long))
        self.register_buffer("pos_features", torch.tensor(pos_features, dtype=torch.float32))

    def forward(self, stickers: torch.Tensor) -> torch.Tensor:
        """stickers: (B, 96) integer tensor of color indices 0-5.
        Returns: (B,) predicted distances."""
        B = stickers.size(0)
        color_emb = self.color_embed(stickers)
        face_emb = self.face_embed(self.face_indices).unsqueeze(0).expand(B, -1, -1)
        pos_emb = self.pos_proj(self.pos_features).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([color_emb, face_emb, pos_emb], dim=-1)
        x = self.input_proj(x)
        for layer in self.mixer_layers:
            x = layer(x)
        x = self.final_norm(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 3x3 distance net (for loading stage 3 checkpoint)
# ---------------------------------------------------------------------------

class DistanceNet3x3(nn.Module):
    """Same architecture as value_search_3x3.py for checkpoint loading."""

    def __init__(
        self,
        n_stickers: int = 54,
        n_colors: int = 6,
        n_faces: int = 6,
        color_embed_dim: int = 64,
        face_embed_dim: int = 32,
        pos_dim: int = 4,
        d_model: int = 512,
        n_layers: int = 6,
        token_mix_dim: int = 256,
        channel_mix_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_stickers = n_stickers
        self.d_model = d_model

        self.color_embed = nn.Embedding(n_colors, color_embed_dim)
        self.face_embed = nn.Embedding(n_faces, face_embed_dim)
        self.pos_proj = nn.Linear(2, pos_dim)

        input_dim = color_embed_dim + face_embed_dim + pos_dim
        self.input_proj = nn.Linear(input_dim, d_model)

        self.mixer_layers = nn.ModuleList([
            MixerLayer(n_stickers, d_model, token_mix_dim, channel_mix_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        face_indices = []
        pos_features = []
        for face_idx, face in enumerate(FACE_ORDER):
            for r in range(3):
                for c in range(3):
                    face_indices.append(face_idx)
                    pos_features.append([r / 1.0 - 1.0, c / 1.0 - 1.0])
        self.register_buffer("face_indices", torch.tensor(face_indices, dtype=torch.long))
        self.register_buffer("pos_features", torch.tensor(pos_features, dtype=torch.float32))

    def forward(self, stickers: torch.Tensor) -> torch.Tensor:
        B = stickers.size(0)
        color_emb = self.color_embed(stickers)
        face_emb = self.face_embed(self.face_indices).unsqueeze(0).expand(B, -1, -1)
        pos_emb = self.pos_proj(self.pos_features).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([color_emb, face_emb, pos_emb], dim=-1)
        x = self.input_proj(x)
        for layer in self.mixer_layers:
            x = layer(x)
        x = self.final_norm(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 4x4 -> 3x3 reduction mapping
# ---------------------------------------------------------------------------

def convert_4x4_to_3x3(cube_4x4: Cube) -> Cube:
    """Convert a reduced 4x4 (centers solved + edges paired) to a 3x3 Cube.

    The 3x3 representation uses:
    - Center: the uniform center color of each face
    - Edges: the color of one wing from each paired edge pair
    - Corners: the corner sticker colors (same positions as 3x3)

    Returns a 3x3 Cube object that can be solved with the 3x3 solver.
    """
    cube_3x3 = Cube(3)

    for face in FACE_ORDER:
        grid_4 = cube_4x4.face_grid(face)
        grid_3 = cube_3x3.face_grid(face)

        # Center of 3x3 = center of 4x4 (any of the 4 center stickers, they're uniform)
        center_color = grid_4[1][1]

        # Corners of 3x3 = corners of 4x4
        # 3x3 corners: (0,0), (0,2), (2,0), (2,2)
        # 4x4 corners: (0,0), (0,3), (3,0), (3,3)
        corners_3 = [(0, 0), (0, 2), (2, 0), (2, 2)]
        corners_4 = [(0, 0), (0, 3), (3, 0), (3, 3)]

        # Edges of 3x3 = edges of 4x4 (pick one wing from each pair)
        # 3x3 edges: (0,1), (1,0), (1,2), (2,1)
        # 4x4 edge wings (picking the one closer to corner 0,0 for top/left):
        #   top edge: (0,1) or (0,2) -- pick (0,1)
        #   left edge: (1,0) or (2,0) -- pick (1,0)
        #   right edge: (1,3) or (2,3) -- pick (1,3)
        #   bottom edge: (3,1) or (3,2) -- pick (3,1)
        edges_3 = [(0, 1), (1, 0), (1, 2), (2, 1)]
        edges_4 = [(0, 1), (1, 0), (1, 3), (3, 1)]

        # Build the sticker mapping using the cube's internal representation
        # We need to set stickers on the 3x3 cube directly
        # Since face_grid returns a view, we need to manipulate stickers dict
        pass

    # Direct sticker-by-sticker mapping via face grids
    # We'll build the 3x3 state by reading 4x4 face grids
    # and constructing equivalent 3x3 sticker colors
    new_cube = Cube(3)

    for face in FACE_ORDER:
        g4 = cube_4x4.face_grid(face)
        # Map: 3x3 position -> 4x4 position
        # (0,0) -> (0,0)  corner
        # (0,1) -> (0,1)  edge (wing)
        # (0,2) -> (0,3)  corner
        # (1,0) -> (1,0)  edge (wing)
        # (1,1) -> (1,1)  center
        # (1,2) -> (1,3)  edge (wing)  -- actually (2,3) is the other wing
        # (2,0) -> (3,0)  corner
        # (2,1) -> (3,1)  edge (wing)
        # (2,2) -> (3,3)  corner
        mapping = {
            (0, 0): (0, 0),
            (0, 1): (0, 1),  # top edge, left wing
            (0, 2): (0, 3),
            (1, 0): (1, 0),  # left edge, top wing
            (1, 1): (1, 1),  # center
            (1, 2): (1, 3),  # right edge, top wing
            (2, 0): (3, 0),
            (2, 1): (3, 1),  # bottom edge, left wing
            (2, 2): (3, 3),
        }

        # Set stickers on 3x3 by finding matching (position, normal) keys
        g3 = new_cube.face_grid(face)
        limit_3 = new_cube.limit  # = 2 for 3x3
        limit_4 = cube_4x4.limit  # = 3 for 4x4

        for (r3, c3), (r4, c4) in mapping.items():
            color = g4[r4][c4]
            # We need to set this on the 3x3 cube's stickers dict
            # Find the sticker in the 3x3 that corresponds to face, row r3, col c3
            # face_grid returns colors in order, so we need the position/normal
            # Actually, let's just set via the grid index approach
            # The simplest way: build a flat color array and reconstruct
            pass

    # Simpler approach: build the 3x3 state as a color string and construct
    # Actually, the cleanest way is to directly set sticker dict entries
    # Let's use the known face_grid -> sticker mapping

    # For each face, face_grid(face) returns an NxN grid.
    # We need to find the sticker (position, normal) for each grid cell.
    # We'll iterate the 3x3 cube's stickers and map them.

    # Build lookup: for each face, grid position -> (position, normal) key in the cube
    def build_grid_to_sticker_map(cube: Cube) -> dict:
        """Map (face, row, col) -> (position, normal) sticker key."""
        result = {}
        n = cube.size
        limit = cube.limit
        for face in FACE_ORDER:
            grid = cube.face_grid(face)
            # We need to figure out which stickers correspond to which grid cells
            # face_grid implementation iterates stickers grouped by face normal
            pass
        return result

    # Even simpler: we know face_grid returns a list of lists of colors.
    # We just need to SET colors in the 3x3 cube to match the 4x4 mapping.
    # The most robust way: iterate ALL stickers in the 3x3 cube,
    # determine their face and grid position, then look up the corresponding
    # 4x4 color.

    # Let's do it by finding sticker keys from face_grid ordering.
    # face_grid sorts by the two non-normal coordinates.
    # For face U (normal (0,1,0) i.e. +y), it sorts by (-z, x) for a left-to-right,
    # top-to-bottom grid. But the exact ordering depends on the implementation.

    # SAFEST approach: just rebuild using the cube's own API.
    # We'll apply the color mapping to the raw sticker dict.

    # Map each 3x3 sticker to the corresponding 4x4 sticker color.
    for (pos3, norm3), color3 in list(new_cube.stickers.items()):
        # Determine which face this sticker is on
        # The face is determined by the normal direction
        # Convert pos3 coordinates to 4x4 coordinates:
        # 3x3 limit=2, coords in {-2, 0, 2}
        # 4x4 limit=3, coords in {-3, -1, 1, 3}
        # Mapping: -2 -> -3, 0 -> -1 or 1, 2 -> 3
        # But 0 maps to EITHER -1 or 1 depending on context (edge vs center)

        # Actually, for the face sticker projection:
        # The normal tells us which face. The other two coords give row/col.
        # 3x3 grid positions: -2, 0, 2 (3 values)
        # 4x4 grid positions: -3, -1, 1, 3 (4 values)
        # Mapping for face stickers:
        #   3x3 coord -2 -> 4x4 coord -3  (corner/outer)
        #   3x3 coord  0 -> 4x4 coord -1  (edge/center, "first" inner)
        #   3x3 coord  2 -> 4x4 coord  3  (corner/outer)

        pos4 = []
        for i, c in enumerate(pos3):
            if norm3[i] != 0:
                # Normal direction: scale from limit 2 to limit 3
                pos4.append(3 if c > 0 else -3)
            else:
                # Grid direction
                if c == -2:
                    pos4.append(-3)
                elif c == 0:
                    pos4.append(-1)
                elif c == 2:
                    pos4.append(3)
                else:
                    pos4.append(c)  # shouldn't happen
        pos4 = tuple(pos4)
        norm4 = norm3  # same normal

        if (pos4, norm4) in cube_4x4.stickers:
            new_cube.stickers[(pos3, norm3)] = cube_4x4.stickers[(pos4, norm4)]

    return new_cube


# ---------------------------------------------------------------------------
# Self-supervised data generation for each stage
# ---------------------------------------------------------------------------

def sample_stage1_endpoints(
    batch_size: int,
    max_depth: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate training data for stage 1 (center solving).

    Start from solved cube (centers solved), apply random moves,
    target = 6 - centers_solved_count (how many centers are broken).
    """
    all_indices = []
    all_targets = []

    # Include solved state with target 0
    cube = Cube(4)
    idx = []
    for face in FACE_ORDER:
        for row in cube.face_grid(face):
            for color in row:
                idx.append(_COLOR_TO_IDX[color])
    all_indices.append(idx)
    all_targets.append(0.0)

    for _ in range(batch_size):
        d = rng.randint(1, max_depth)
        cube = Cube(4)
        last_move_idx = None

        for step in range(d):
            valid = get_valid_move_indices_4x4(last_move_idx)
            move_idx = rng.choice(valid)
            cube.apply_move(ALL_MOVES_4x4[move_idx])
            last_move_idx = move_idx

        # Target: number of unsolved center faces (0-6)
        # Use walk distance as target (more informative than just broken count)
        # Blend: use walk distance but clamp by actual broken count
        broken = 6 - centers_solved_count_444(cube)
        target = float(d) if broken > 0 else 0.0

        idx = []
        for face in FACE_ORDER:
            for row in cube.face_grid(face):
                for color in row:
                    idx.append(_COLOR_TO_IDX[color])
        all_indices.append(idx)
        all_targets.append(target)

    states = torch.tensor(all_indices, dtype=torch.long)
    targets = torch.tensor(all_targets, dtype=torch.float32)
    return states, targets


def _make_reduced_state(rng: random.Random, n_scramble_moves: int = 20) -> Cube:
    """Create a 'reduced' 4x4 state: centers solved + edges paired, corners scrambled.

    Starting from solved, apply only outer-face moves (width=1) which scramble
    corners but preserve center blocks and edge pairing on a 4x4.
    """
    cube = Cube(4)
    last_move_idx = None
    for _ in range(n_scramble_moves):
        # Filter to only outer moves that are also valid (no redundant turns)
        valid = get_valid_move_indices_4x4(last_move_idx)
        outer_valid = [i for i in valid if i in _OUTER_MOVE_SET]
        move_idx = rng.choice(outer_valid)
        cube.apply_move(ALL_MOVES_4x4[move_idx])
        last_move_idx = move_idx
    return cube


# Precomputed set for fast lookup
_OUTER_MOVE_SET = set(OUTER_MOVE_INDICES)


def sample_stage2_endpoints(
    batch_size: int,
    max_depth: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate training data for stage 2 (edge pairing) via reverse walks.

    Strategy (stage-specific reverse training):
      1. Generate 'reduced' states as goals: centers solved + edges paired
         (corners may be scrambled). These are created by applying only
         outer-face moves (width=1) to a solved cube.
      2. From each reduced state, apply K random moves (ANY move including
         wide moves that break edge pairing). K is sampled from [1, max_depth].
         The label is K (distance from edge-paired state).
      3. Include some reduced states with K=0 (target=0) so the model learns
         the goal state.

    This ensures the training data covers states with partially paired edges
    (the states beam search actually visits), not just fully-paired or
    fully-broken extremes.
    """
    all_indices = []
    all_targets = []

    # ~10% of batch are goal states (K=0) so the model learns target=0
    n_goals = max(1, batch_size // 10)

    for _ in range(n_goals):
        cube = _make_reduced_state(rng)
        idx = []
        for face in FACE_ORDER:
            for row in cube.face_grid(face):
                for color in row:
                    idx.append(_COLOR_TO_IDX[color])
        all_indices.append(idx)
        all_targets.append(0.0)

    # Remaining samples: reverse walks from reduced states
    for _ in range(batch_size - n_goals):
        cube = _make_reduced_state(rng)
        d = rng.randint(1, max_depth)
        last_move_idx = None

        for step in range(d):
            valid = get_valid_move_indices_4x4(last_move_idx)
            move_idx = rng.choice(valid)
            cube.apply_move(ALL_MOVES_4x4[move_idx])
            last_move_idx = move_idx

        idx = []
        for face in FACE_ORDER:
            for row in cube.face_grid(face):
                for color in row:
                    idx.append(_COLOR_TO_IDX[color])
        all_indices.append(idx)
        all_targets.append(float(d))

    states = torch.tensor(all_indices, dtype=torch.long)
    targets = torch.tensor(all_targets, dtype=torch.float32)
    return states, targets


# ---------------------------------------------------------------------------
# DeepCubeA-style bootstrap for stage heuristics
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_stage_bootstrap_targets(
    model: StageDistanceNet,
    batch_size: int,
    max_depth: int,
    rng: random.Random,
    stage: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepCubeA-style bootstrap targets for a stage heuristic.

    1. Random walk endpoints from solved
    2. Expand all 36 children
    3. Target = 1 + min(model(children))
    4. Stage-done states get target = 0
    """
    model.eval()

    is_done_fn = stage1_is_done if stage == 1 else stage2_is_done

    cubes = []
    for _ in range(batch_size):
        d = rng.randint(1, max_depth)
        cube = Cube(4)
        last_move_idx = None
        for step in range(d):
            valid = get_valid_move_indices_4x4(last_move_idx)
            move_idx = rng.choice(valid)
            cube.apply_move(ALL_MOVES_4x4[move_idx])
            last_move_idx = move_idx
        cubes.append(cube)

    # Include some "solved" states (stage goal met)
    n_solved = max(1, batch_size // 20)
    for _ in range(n_solved):
        cubes.append(Cube(4))

    all_states = encode_cube_batch_4x4(cubes).to(device)
    B = len(cubes)

    # Expand children
    child_cubes = []
    for cube in cubes:
        for move in ALL_MOVES_4x4:
            child = cube.copy()
            child.apply_move(move)
            child_cubes.append(child)

    child_states = encode_cube_batch_4x4(child_cubes).to(device)

    chunk_size = 4096
    all_child_values = []
    for i in range(0, len(child_cubes), chunk_size):
        chunk = child_states[i:i + chunk_size]
        vals = model(chunk)
        all_child_values.append(vals)
    child_values = torch.cat(all_child_values, dim=0)
    child_values = child_values.view(B, N_MOVES_4x4)

    targets = 1.0 + child_values.min(dim=1).values

    # Override: stage-done states get target 0
    for i, cube in enumerate(cubes):
        if is_done_fn(cube):
            targets[i] = 0.0

    targets = targets.clamp(min=0.0)
    return all_states, targets


# ---------------------------------------------------------------------------
# Beam search for stages 1 and 2
# ---------------------------------------------------------------------------

@dataclass
class BeamItem:
    cube: Cube
    history: list[int]  # move indices
    last_move_idx: Optional[int]


@torch.no_grad()
def beam_search_stage(
    model: StageDistanceNet,
    cube: Cube,
    is_done_fn: Callable[[Cube], bool],
    beam_width: int = 64,
    max_steps: int = 300,
    device: str = "cuda",
    allowed_move_indices: Optional[list[int]] = None,
) -> tuple[bool, list[Move], int]:
    """Beam search guided by stage heuristic.

    Returns (success, move_list, nodes_expanded).
    """
    model.eval()

    if is_done_fn(cube):
        return True, [], 0

    beam = [BeamItem(cube=cube.copy(), history=[], last_move_idx=None)]
    visited: set[str] = {cube.to_kociemba_string()}
    nodes_expanded = 0

    for step in range(max_steps):
        candidates: list[tuple[Cube, list[int], int, str]] = []
        for item in beam:
            if is_done_fn(item.cube):
                moves = [ALL_MOVES_4x4[i] for i in item.history]
                return True, moves, nodes_expanded

            valid_moves = get_valid_move_indices_4x4(item.last_move_idx)
            if allowed_move_indices is not None:
                valid_moves = [m for m in valid_moves if m in allowed_move_indices]

            for move_idx in valid_moves:
                move = ALL_MOVES_4x4[move_idx]
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

        # Score candidates
        states = encode_cube_batch_4x4([c for c, _, _, _ in candidates]).to(device)
        chunk_size = 4096
        all_scores = []
        for i in range(0, len(candidates), chunk_size):
            chunk = states[i:i + chunk_size]
            scores = model(chunk).cpu()
            all_scores.append(scores)
        pred_distances = torch.cat(all_scores, dim=0)

        scored = []
        for i, (c, hist, last_idx, key) in enumerate(candidates):
            h = pred_distances[i].item()
            scored.append((h, i, c, hist, last_idx, key))

        scored.sort(key=lambda x: x[0])

        beam = []
        for h, i, c, hist, last_idx, key in scored:
            if is_done_fn(c):
                moves = [ALL_MOVES_4x4[mi] for mi in hist]
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
# Stage 3: Use 3x3 solver
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_search_3x3(
    model: DistanceNet3x3,
    cube: Cube,
    beam_width: int = 32,
    max_steps: int = 200,
    device: str = "cuda",
) -> tuple[bool, list[Move], int]:
    """Beam search on a 3x3 cube using the 3x3 distance network."""
    model.eval()

    if cube.is_solved():
        return True, [], 0

    beam = [BeamItem(cube=cube.copy(), history=[], last_move_idx=None)]
    visited: set[str] = {cube.to_kociemba_string()}
    nodes_expanded = 0

    for step in range(max_steps):
        candidates: list[tuple[Cube, list[int], int, str]] = []
        for item in beam:
            if item.cube.is_solved():
                moves = [ALL_MOVES_3x3[i] for i in item.history]
                return True, moves, nodes_expanded

            valid_moves = get_valid_move_indices_3x3(item.last_move_idx)
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

        states = encode_cube_batch_3x3([c for c, _, _, _ in candidates]).to(device)
        chunk_size = 4096
        all_scores = []
        for i in range(0, len(candidates), chunk_size):
            chunk = states[i:i + chunk_size]
            scores = model(chunk).cpu()
            all_scores.append(scores)
        pred_distances = torch.cat(all_scores, dim=0)

        scored = []
        for i, (c, hist, last_idx, key) in enumerate(candidates):
            h = pred_distances[i].item()
            scored.append((h, i, c, hist, last_idx, key))

        scored.sort(key=lambda x: x[0])

        beam = []
        for h, i, c, hist, last_idx, key in scored:
            if c.is_solved():
                moves = [ALL_MOVES_3x3[mi] for mi in hist]
                return True, moves, nodes_expanded
            if len(beam) >= beam_width:
                continue
            if key not in visited:
                visited.add(key)
                beam.append(BeamItem(cube=c, history=hist, last_move_idx=last_idx))

        if not beam:
            break

    return False, [], nodes_expanded


def solve_stage3_on_4x4(
    model_3x3: DistanceNet3x3,
    cube_4x4: Cube,
    beam_width: int = 32,
    max_steps: int = 200,
    device: str = "cuda",
) -> tuple[bool, list[Move], int]:
    """Solve a reduced 4x4 by converting to 3x3, solving, then mapping moves back.

    Since a reduced 4x4 (centers solved + edges paired) behaves like a 3x3,
    outer-layer moves on the 4x4 correspond directly to 3x3 moves.
    We convert the state, solve the 3x3, and return the same moves
    (which are valid outer-layer moves on the 4x4).
    """
    cube_3x3 = convert_4x4_to_3x3(cube_4x4)
    solved, moves_3x3, nodes = beam_search_3x3(
        model_3x3, cube_3x3, beam_width=beam_width,
        max_steps=max_steps, device=device,
    )

    if solved:
        # 3x3 moves map directly to 4x4 outer-layer moves
        # (same face, same turns, width=1 on 4x4)
        moves_4x4 = []
        for m in moves_3x3:
            moves_4x4.append(Move(face=m.face, depth=1, width=1, turns=m.turns))
        return True, moves_4x4, nodes
    return False, [], nodes


# ---------------------------------------------------------------------------
# Full hierarchical solve pipeline
# ---------------------------------------------------------------------------

def load_stage_model(
    stage: int,
    checkpoint_dir: str,
    device: str = "cuda",
    d_model: int = 256,
    n_layers: int = 5,
    token_mix_dim: int = 128,
    channel_mix_dim: int = 512,
) -> StageDistanceNet:
    """Load a trained stage heuristic model."""
    model = StageDistanceNet(
        d_model=d_model,
        n_layers=n_layers,
        token_mix_dim=token_mix_dim,
        channel_mix_dim=channel_mix_dim,
    ).to(device)

    ckpt_path = os.path.join(checkpoint_dir, "final.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded stage {stage} model from {ckpt_path}")
    else:
        print(f"WARNING: No checkpoint found for stage {stage} at {checkpoint_dir}")
    return model


def load_3x3_model(
    checkpoint_dir: str = "checkpoints/value_3x3_v2",
    device: str = "cuda",
) -> DistanceNet3x3:
    """Load the trained 3x3 solver model."""
    model = DistanceNet3x3(
        d_model=512,
        n_layers=6,
        token_mix_dim=256,
        channel_mix_dim=2048,
    ).to(device)

    ckpt_path = os.path.join(checkpoint_dir, "final.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded 3x3 model from {ckpt_path}")
    else:
        print(f"WARNING: No 3x3 checkpoint found at {checkpoint_dir}")
    return model


@torch.no_grad()
def full_solve_4x4(
    model_s1: StageDistanceNet,
    model_s2: StageDistanceNet,
    model_3x3: DistanceNet3x3,
    cube: Cube,
    beam_width_s1: int = 64,
    beam_width_s2: int = 64,
    beam_width_s3: int = 32,
    max_steps_s1: int = 300,
    max_steps_s2: int = 300,
    max_steps_s3: int = 200,
    device: str = "cuda",
    verbose: bool = False,
) -> tuple[bool, list[Move], dict]:
    """Full hierarchical solve: stage1 -> stage2 -> stage3.

    Returns (solved, all_moves, info_dict).
    """
    all_moves: list[Move] = []
    info = {
        "stage1_solved": False,
        "stage2_solved": False,
        "stage3_solved": False,
        "stage1_moves": 0,
        "stage2_moves": 0,
        "stage3_moves": 0,
        "stage1_nodes": 0,
        "stage2_nodes": 0,
        "stage3_nodes": 0,
    }

    current = cube.copy()

    # Stage 1: Solve centers
    if verbose:
        print(f"  Stage 1: centers_solved={centers_solved_count_444(current)}/6")
    if not centers_done_444(current):
        ok, moves, nodes = beam_search_stage(
            model_s1, current, stage1_is_done,
            beam_width=beam_width_s1, max_steps=max_steps_s1, device=device,
        )
        info["stage1_nodes"] = nodes
        if ok:
            info["stage1_solved"] = True
            info["stage1_moves"] = len(moves)
            for m in moves:
                current.apply_move(m)
            all_moves.extend(moves)
            if verbose:
                print(f"  Stage 1 solved in {len(moves)} moves ({nodes} nodes)")
        else:
            if verbose:
                print(f"  Stage 1 FAILED ({nodes} nodes)")
            return False, all_moves, info
    else:
        info["stage1_solved"] = True
        if verbose:
            print("  Stage 1 already done")

    # Stage 2: Pair edges (preserve centers)
    if verbose:
        print(f"  Stage 2: paired_edges={paired_edge_count(current)}/12, "
              f"centers_ok={centers_done_444(current)}")
    if not edges_paired(current):
        ok, moves, nodes = beam_search_stage(
            model_s2, current, stage2_is_done,
            beam_width=beam_width_s2, max_steps=max_steps_s2, device=device,
        )
        info["stage2_nodes"] = nodes
        if ok:
            info["stage2_solved"] = True
            info["stage2_moves"] = len(moves)
            for m in moves:
                current.apply_move(m)
            all_moves.extend(moves)
            if verbose:
                print(f"  Stage 2 solved in {len(moves)} moves ({nodes} nodes)")
        else:
            if verbose:
                print(f"  Stage 2 FAILED ({nodes} nodes)")
            return False, all_moves, info
    else:
        info["stage2_solved"] = True
        if verbose:
            print("  Stage 2 already done")

    # Verify reduction is complete
    if not (centers_done_444(current) and edges_paired(current)):
        if verbose:
            print("  WARNING: Reduction incomplete after stage 2!")
        return False, all_moves, info

    # Stage 3: Solve as 3x3
    if verbose:
        print(f"  Stage 3: solving as 3x3...")
    if current.is_solved():
        info["stage3_solved"] = True
        if verbose:
            print("  Already solved!")
        return True, all_moves, info

    ok, moves_s3, nodes = solve_stage3_on_4x4(
        model_3x3, current,
        beam_width=beam_width_s3, max_steps=max_steps_s3, device=device,
    )
    info["stage3_nodes"] = nodes
    if ok:
        info["stage3_solved"] = True
        info["stage3_moves"] = len(moves_s3)
        for m in moves_s3:
            current.apply_move(m)
        all_moves.extend(moves_s3)
        if verbose:
            print(f"  Stage 3 solved in {len(moves_s3)} moves ({nodes} nodes)")
    else:
        if verbose:
            print(f"  Stage 3 FAILED ({nodes} nodes)")
        return False, all_moves, info

    # Verify
    solved = current.is_solved()
    if verbose:
        print(f"  Final: solved={solved}, total_moves={len(all_moves)}")
    return solved, all_moves, info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_pipeline(
    model_s1: StageDistanceNet,
    model_s2: StageDistanceNet,
    model_3x3: DistanceNet3x3,
    num_cubes: int = 50,
    scramble_length: int = 40,
    beam_width_s1: int = 64,
    beam_width_s2: int = 64,
    beam_width_s3: int = 32,
    max_steps_s1: int = 300,
    max_steps_s2: int = 300,
    max_steps_s3: int = 200,
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Evaluate the full hierarchical pipeline on random scrambles."""
    rng = random.Random(seed)
    solved_count = 0
    s1_count = 0
    s2_count = 0
    s3_count = 0
    total_moves = 0
    total_nodes = 0

    for i in range(num_cubes):
        scramble = random_scramble(4, scramble_length, rng, max_depth=1, max_width=2)
        cube = Cube(4)
        cube.apply_moves(scramble)

        if verbose:
            print(f"\nCube {i+1}/{num_cubes}:")

        ok, moves, info = full_solve_4x4(
            model_s1, model_s2, model_3x3, cube,
            beam_width_s1=beam_width_s1, beam_width_s2=beam_width_s2,
            beam_width_s3=beam_width_s3,
            max_steps_s1=max_steps_s1, max_steps_s2=max_steps_s2,
            max_steps_s3=max_steps_s3,
            device=device, verbose=verbose,
        )

        if info["stage1_solved"]:
            s1_count += 1
        if info["stage2_solved"]:
            s2_count += 1
        if info["stage3_solved"]:
            s3_count += 1
        if ok:
            solved_count += 1
            total_moves += len(moves)
        total_nodes += info["stage1_nodes"] + info["stage2_nodes"] + info["stage3_nodes"]

    n = num_cubes
    return {
        "solve_rate": solved_count / n,
        "solved": solved_count,
        "total": n,
        "stage1_rate": s1_count / n,
        "stage2_rate": s2_count / n,
        "stage3_rate": s3_count / n,
        "mean_solution_length": total_moves / max(1, solved_count),
        "mean_nodes": total_nodes / n,
    }


# ---------------------------------------------------------------------------
# Stage-specific evaluation (for training)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_stage(
    model: StageDistanceNet,
    stage: int,
    num_cubes: int = 50,
    scramble_length: int = 20,
    beam_width: int = 64,
    max_steps: int = 300,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """Evaluate a single stage heuristic on random scrambles."""
    rng = random.Random(seed)
    is_done_fn = stage1_is_done if stage == 1 else stage2_is_done

    solved_count = 0
    total_moves = 0
    total_nodes = 0

    for i in range(num_cubes):
        if stage == 2:
            # Stage 2 eval: start from reduced state (centers solved), then break edges
            cube = _make_reduced_state(rng)
            moves_all = [Move(face=f, depth=1, width=w, turns=t)
                         for f in "URFDLB" for w in [1, 2] for t in [1, -1, 2]]
            for _ in range(scramble_length):
                cube.apply_move(rng.choice(moves_all))
        else:
            # Stage 1 eval: fully scrambled cube
            scramble = random_scramble(4, scramble_length, rng, max_depth=1, max_width=2)
            cube = Cube(4)
            cube.apply_moves(scramble)

        ok, moves, nodes = beam_search_stage(
            model, cube, is_done_fn,
            beam_width=beam_width, max_steps=max_steps, device=device,
        )

        if ok:
            solved_count += 1
            total_moves += len(moves)
        total_nodes += nodes

    n = num_cubes
    return {
        "solve_rate": solved_count / n,
        "solved": solved_count,
        "total": n,
        "mean_solution_length": total_moves / max(1, solved_count),
        "mean_nodes": total_nodes / n,
    }


# ---------------------------------------------------------------------------
# Training loop for a single stage
# ---------------------------------------------------------------------------

def train_stage(args):
    """Train a single stage heuristic."""
    stage = args.stage
    assert stage in (1, 2), f"Can only train stages 1 or 2, got {stage}"

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

    model = StageDistanceNet(
        d_model=args.d_model,
        n_layers=args.n_layers,
        token_mix_dim=args.token_mix_dim,
        channel_mix_dim=args.channel_mix_dim,
        dropout=args.dropout,
    ).to(device)

    print(f"Stage {stage} model parameters: {model.num_params():,}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, args.total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if use_wandb:
        wandb.init(
            project="rubiks-4x4-hierarchical",
            config=vars(args),
            name=f"stage{stage}-{args.d_model}d-{args.n_layers}L",
        )

    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    rng = random.Random(args.seed)
    start_time = time.time()
    max_seconds = args.hours * 3600
    global_step = 0

    curr_max_depth = args.curriculum_start
    bootstrap_start_step = args.bootstrap_after

    # Data generation function
    sample_fn = sample_stage1_endpoints if stage == 1 else sample_stage2_endpoints

    print(f"Training stage {stage} for up to {args.hours} hours", flush=True)
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

        # Curriculum
        progress = min(1.0, global_step / max(1, args.curriculum_ramp_steps))
        curr_max_depth = int(
            args.curriculum_start + progress * (args.curriculum_end - args.curriculum_start)
        )
        curr_max_depth = max(args.curriculum_start, min(args.curriculum_end, curr_max_depth))

        use_bootstrap = global_step >= bootstrap_start_step

        if use_bootstrap:
            states, targets = generate_stage_bootstrap_targets(
                model, args.batch_size, curr_max_depth, rng, stage, device
            )
            model.train()
        else:
            states, targets = sample_fn(args.batch_size, curr_max_depth, rng)
            states = states.to(device)
            targets = targets.to(device)

        predictions = model(states)
        loss = F.smooth_l1_loss(predictions, targets)

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

        # Evaluation
        if global_step % args.eval_interval == 0:
            print(f"\n--- Stage {stage} Evaluation at step {global_step} ---")
            for bw in args.eval_beam_widths:
                results = evaluate_stage(
                    model, stage,
                    num_cubes=args.eval_num_cubes,
                    scramble_length=args.eval_scramble_length,
                    beam_width=bw,
                    max_steps=args.eval_max_steps,
                    device=device,
                    seed=args.eval_seed,
                )
                print(
                    f"  beam_width={bw:4d}: stage{stage}_rate={results['solve_rate']:.3f} "
                    f"({results['solved']}/{results['total']}) "
                    f"mean_sol_len={results['mean_solution_length']:.1f} "
                    f"mean_nodes={results['mean_nodes']:.0f}"
                )
                if use_wandb:
                    wandb.log({
                        f"eval/stage{stage}_rate_bw{bw}": results["solve_rate"],
                        f"eval/mean_sol_len_bw{bw}": results["mean_solution_length"],
                        f"eval/mean_nodes_bw{bw}": results["mean_nodes"],
                    }, step=global_step)
            print(flush=True)

        # Checkpoint
        if global_step % args.save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
            ckpt_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": global_step,
                "curr_max_depth": curr_max_depth,
                "args": vars(args),
                "stage": stage,
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
        "stage": stage,
    }, final_path)
    print(f"Saved final checkpoint: {final_path}")

    # Final evaluation
    print(f"\n=== Stage {stage} Final Evaluation ===")
    for bw in [1, 8, 32, 64, 128]:
        results = evaluate_stage(
            model, stage,
            num_cubes=100,
            scramble_length=args.eval_scramble_length,
            beam_width=bw,
            max_steps=300,
            device=device,
            seed=42,
        )
        print(
            f"  beam_width={bw:4d}: stage{stage}_rate={results['solve_rate']:.3f} "
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
    parser = argparse.ArgumentParser(
        description="Hierarchical reduction-based 4x4 Rubik's cube solver"
    )

    # Mode selection
    parser.add_argument("--stage", type=int, default=0,
                        help="Stage to train (1=centers, 2=edges). 0=no training.")
    parser.add_argument("--solve", action="store_true",
                        help="Run full solve pipeline on a random cube")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate full pipeline on random scrambles")

    # Time budget
    parser.add_argument("--hours", type=float, default=4.0,
                        help="Training time budget in hours")

    # Model architecture (per-stage)
    parser.add_argument("--d_model", type=int, default=256,
                        help="Model hidden dimension")
    parser.add_argument("--n_layers", type=int, default=5,
                        help="Number of mixer layers")
    parser.add_argument("--token_mix_dim", type=int, default=128,
                        help="Token mixing MLP hidden dim")
    parser.add_argument("--channel_mix_dim", type=int, default=512,
                        help="Channel mixing MLP hidden dim")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Training
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size")
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
    parser.add_argument("--curriculum_end", type=int, default=30,
                        help="Final max walk depth")
    parser.add_argument("--curriculum_ramp_steps", type=int, default=80_000,
                        help="Steps over which to ramp depth")

    # Bootstrap
    parser.add_argument("--bootstrap_after", type=int, default=20_000,
                        help="Switch to bootstrap after this many steps")

    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=5_000,
                        help="Steps between evaluations")
    parser.add_argument("--eval_num_cubes", type=int, default=30,
                        help="Cubes per eval")
    parser.add_argument("--eval_scramble_length", type=int, default=30,
                        help="Scramble length for eval")
    parser.add_argument("--eval_max_steps", type=int, default=300,
                        help="Max beam search steps per eval")
    parser.add_argument("--eval_beam_widths", type=int, nargs="+", default=[8, 32, 64],
                        help="Beam widths for evaluation")
    parser.add_argument("--eval_seed", type=int, default=42,
                        help="Seed for eval scrambles")

    # Pipeline eval settings
    parser.add_argument("--pipeline_num_cubes", type=int, default=50,
                        help="Cubes for pipeline evaluation")
    parser.add_argument("--pipeline_scramble_length", type=int, default=40,
                        help="Scramble length for pipeline evaluation")
    parser.add_argument("--beam_width_s1", type=int, default=64,
                        help="Beam width for stage 1 in pipeline")
    parser.add_argument("--beam_width_s2", type=int, default=64,
                        help="Beam width for stage 2 in pipeline")
    parser.add_argument("--beam_width_s3", type=int, default=32,
                        help="Beam width for stage 3 in pipeline")

    # Logging and saving
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Steps between log prints")
    parser.add_argument("--save_interval", type=int, default=10_000,
                        help="Steps between checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Checkpoint directory (default: auto based on stage)")

    # Checkpoint paths for loading
    parser.add_argument("--stage1_ckpt", type=str,
                        default="checkpoints/value_4x4_stage1",
                        help="Stage 1 checkpoint directory")
    parser.add_argument("--stage2_ckpt", type=str,
                        default="checkpoints/value_4x4_stage2",
                        help="Stage 2 checkpoint directory")
    parser.add_argument("--stage3_ckpt", type=str,
                        default="checkpoints/value_3x3_v2",
                        help="3x3 solver checkpoint directory")

    # System
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")

    args = parser.parse_args()

    # Set default checkpoint dir based on stage
    if args.checkpoint_dir is None:
        if args.stage == 1:
            args.checkpoint_dir = "checkpoints/value_4x4_stage1"
        elif args.stage == 2:
            args.checkpoint_dir = "checkpoints/value_4x4_stage2"
        else:
            args.checkpoint_dir = "checkpoints/value_4x4_hierarchical"

    return args


def main():
    args = parse_args()

    if args.stage in (1, 2):
        # Train a stage heuristic
        train_stage(args)

    elif args.solve:
        # Full solve pipeline: solve one random cube
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        print("Loading models...")
        model_s1 = load_stage_model(1, args.stage1_ckpt, device,
                                     d_model=args.d_model, n_layers=args.n_layers,
                                     token_mix_dim=args.token_mix_dim,
                                     channel_mix_dim=args.channel_mix_dim)
        model_s2 = load_stage_model(2, args.stage2_ckpt, device,
                                     d_model=args.d_model, n_layers=args.n_layers,
                                     token_mix_dim=args.token_mix_dim,
                                     channel_mix_dim=args.channel_mix_dim)
        model_3x3 = load_3x3_model(args.stage3_ckpt, device)

        print("\nGenerating random scramble...")
        rng = random.Random(args.seed)
        scramble = random_scramble(4, args.pipeline_scramble_length, rng,
                                   max_depth=1, max_width=2)
        cube = Cube(4)
        cube.apply_moves(scramble)
        print(f"Scramble: {len(scramble)} moves")
        print(f"Initial state: centers={centers_solved_count_444(cube)}/6, "
              f"edges={paired_edge_count(cube)}/12")

        ok, moves, info = full_solve_4x4(
            model_s1, model_s2, model_3x3, cube,
            beam_width_s1=args.beam_width_s1,
            beam_width_s2=args.beam_width_s2,
            beam_width_s3=args.beam_width_s3,
            device=device, verbose=True,
        )
        print(f"\nResult: {'SOLVED' if ok else 'FAILED'}")
        print(f"Total moves: {len(moves)}")
        print(f"Stage info: {info}")

    elif args.eval:
        # Evaluate full pipeline
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        print("Loading models...")
        model_s1 = load_stage_model(1, args.stage1_ckpt, device,
                                     d_model=args.d_model, n_layers=args.n_layers,
                                     token_mix_dim=args.token_mix_dim,
                                     channel_mix_dim=args.channel_mix_dim)
        model_s2 = load_stage_model(2, args.stage2_ckpt, device,
                                     d_model=args.d_model, n_layers=args.n_layers,
                                     token_mix_dim=args.token_mix_dim,
                                     channel_mix_dim=args.channel_mix_dim)
        model_3x3 = load_3x3_model(args.stage3_ckpt, device)

        print(f"\nEvaluating pipeline on {args.pipeline_num_cubes} cubes "
              f"(scramble_length={args.pipeline_scramble_length})...")
        results = evaluate_pipeline(
            model_s1, model_s2, model_3x3,
            num_cubes=args.pipeline_num_cubes,
            scramble_length=args.pipeline_scramble_length,
            beam_width_s1=args.beam_width_s1,
            beam_width_s2=args.beam_width_s2,
            beam_width_s3=args.beam_width_s3,
            device=device,
            seed=args.eval_seed,
            verbose=True,
        )
        print(f"\n=== Pipeline Results ===")
        print(f"  Overall solve rate: {results['solve_rate']:.3f} "
              f"({results['solved']}/{results['total']})")
        print(f"  Stage 1 (centers) rate: {results['stage1_rate']:.3f}")
        print(f"  Stage 2 (edges)   rate: {results['stage2_rate']:.3f}")
        print(f"  Stage 3 (3x3)     rate: {results['stage3_rate']:.3f}")
        print(f"  Mean solution length: {results['mean_solution_length']:.1f}")
        print(f"  Mean nodes expanded: {results['mean_nodes']:.0f}")

    else:
        print("No mode selected. Use --stage 1/2 to train, --solve to solve, --eval to evaluate.")
        print("Examples:")
        print("  python value_search_4x4_hierarchical.py --stage 1 --hours 4")
        print("  python value_search_4x4_hierarchical.py --stage 2 --hours 4")
        print("  python value_search_4x4_hierarchical.py --eval")
        print("  python value_search_4x4_hierarchical.py --solve")


if __name__ == "__main__":
    main()
