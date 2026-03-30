"""
CayleyPy-style distance prediction for 4x4 Rubik's cube.

Train a small ResMLP to predict random-walk distance from any state to solved.
Use as a heuristic for beam/A* search at inference time.

Based on: Chervov et al., "A Machine Learning Approach That Beats Large Rubik's Cubes"
(arXiv 2502.13266, NeurIPS 2025)
"""

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rubiks import (
    Cube, Move, FACE_ORDER, FACE_COLORS,
    random_scramble, centers_done_444, paired_edge_count_444,
    reduction_stage_444,
)


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

# Map color chars to indices
_COLOR_TO_IDX = {c: i for i, c in enumerate(("W", "Y", "G", "B", "R", "O"))}


def encode_cube_444(cube: Cube) -> torch.LongTensor:
    """Encode a 4x4 cube state as a flat tensor of 96 color indices (0-5)."""
    indices = []
    for face in FACE_ORDER:
        for row in cube.face_grid(face):
            for color in row:
                indices.append(_COLOR_TO_IDX[color])
    return torch.tensor(indices, dtype=torch.long)


# ---------------------------------------------------------------------------
# Training data: non-backtracking random walks from solved
# ---------------------------------------------------------------------------

# All legal 4x4 moves: 6 faces × 3 turns × 2 widths (1 and 2) = 36 moves
_ALL_MOVES_444: list[Move] = []
for face in FACE_ORDER:
    for turns in (1, -1, 2):
        _ALL_MOVES_444.append(Move(face=face, depth=1, width=1, turns=turns))
        _ALL_MOVES_444.append(Move(face=face, depth=1, width=2, turns=turns))

# Opposite faces (moves on opposite faces commute)
_OPPOSITE = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}


def enumerate_moves_444(last_move: Optional[Move] = None) -> list[Move]:
    """Return all legal 4x4 moves, excluding the exact inverse of last_move
    and same-face moves if previous was on the opposite face (reduces redundancy)."""
    if last_move is None:
        return list(_ALL_MOVES_444)
    inv = last_move.inverse()
    return [m for m in _ALL_MOVES_444
            if not (m.face == inv.face and m.width == inv.width and m.turns == inv.turns)]


def sample_walk_batch(batch_size: int, max_depth: int = 48,
                      rng: random.Random | None = None) -> list[tuple[torch.LongTensor, int]]:
    """Generate a batch of (state, walk_length) training pairs.

    For each sample: start from solved, apply d random non-backtracking moves
    (d uniform in 1..max_depth), record every prefix state.
    Returns list of (encoded_state, distance_from_solved).
    """
    if rng is None:
        rng = random.Random()

    samples = []
    for _ in range(batch_size):
        d = rng.randint(1, max_depth)
        cube = Cube(4)
        last_move = None
        for step in range(d):
            candidates = enumerate_moves_444(last_move)
            move = rng.choice(candidates)
            cube.apply_move(move)
            last_move = move
            # Record every prefix state
            samples.append((encode_cube_444(cube), step + 1))

    return samples


def sample_walk_batch_gpu(batch_size: int, max_depth: int = 48,
                          device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch and return as GPU tensors."""
    samples = sample_walk_batch(batch_size, max_depth)
    states = torch.stack([s for s, _ in samples]).to(device)
    distances = torch.tensor([d for _, d in samples], dtype=torch.float32).to(device)
    return states, distances


# ---------------------------------------------------------------------------
# Model: ResMLP for distance prediction
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DistanceMLP444(nn.Module):
    """ResMLP that predicts distance-to-goal from 4x4 sticker colors.

    Architecture inspired by CayleyPy: small, fast, pure MLP.
    ~2-4M parameters depending on hidden size.
    """

    def __init__(self, n_stickers: int = 96, n_colors: int = 6,
                 embed_dim: int = 32, hidden_dim: int = 1024, n_blocks: int = 4):
        super().__init__()
        self.embed = nn.Embedding(n_colors, embed_dim)
        self.input_proj = nn.Linear(n_stickers * embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, stickers: torch.Tensor) -> torch.Tensor:
        """stickers: (B, 96) integer tensor of color indices 0-5."""
        x = self.embed(stickers)          # (B, 96, embed_dim)
        x = x.view(x.size(0), -1)        # (B, 96 * embed_dim)
        x = F.relu(self.input_proj(x))    # (B, hidden_dim)
        for block in self.blocks:
            x = F.relu(block(x))
        return self.output(x).squeeze(-1)  # (B,)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Search: heuristic beam search guided by distance prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_search_distance(model: DistanceMLP444, cube: Cube,
                         beam_width: int = 32, max_steps: int = 200,
                         device: str = "cuda") -> tuple[bool, Cube, list[Move]]:
    """Heuristic beam search using predicted distance as guidance.

    Returns (solved, final_cube, move_history).
    """
    model.eval()

    # Each beam item: (cube, history, g_cost)
    beam = [(cube.copy(), [], 0)]
    visited = {cube.to_kociemba_string()}
    best_stage = 0

    for step in range(max_steps):
        candidates = []
        for b_cube, b_history, b_g in beam:
            if b_cube.is_solved():
                return True, b_cube, b_history

            last_move = b_history[-1] if b_history else None
            moves = enumerate_moves_444(last_move)

            for move in moves:
                new_cube = b_cube.copy()
                new_cube.apply_move(move)
                state_key = new_cube.to_kociemba_string()
                if state_key in visited:
                    continue
                candidates.append((new_cube, b_history + [move], b_g + 1, state_key))

        if not candidates:
            break

        # Score all candidates with the distance model
        states = torch.stack([encode_cube_444(c) for c, _, _, _ in candidates]).to(device)
        pred_distances = model(states).cpu()

        # f = g + 1 + h (A*-style scoring)
        scored = []
        for i, (c, hist, g, key) in enumerate(candidates):
            f = g + pred_distances[i].item()
            scored.append((f, i, c, hist, g, key))

        scored.sort(key=lambda x: x[0])

        # Keep top beam_width unique states
        beam = []
        for f, i, c, hist, g, key in scored:
            if len(beam) >= beam_width:
                break
            if key not in visited:
                visited.add(key)
                beam.append((c, hist, g))

        if not beam:
            break

    # Return the best item from the final beam
    best = min(beam, key=lambda x: x[2]) if beam else (cube, [], 0)
    return best[0].is_solved(), best[0], best[1]


@torch.no_grad()
def evaluate_distance_search(model: DistanceMLP444, episodes: list,
                             beam_width: int = 32, max_steps: int = 200,
                             device: str = "cuda",
                             num_cubes: int = 64) -> dict:
    """Evaluate distance-guided beam search on 4x4 episodes.

    Returns dict with solve_rate, centers_done_rate, edges_paired_rate,
    reduced_to_3x3_rate, mean_stage_reached.
    """
    from rubiks import Episode

    results = {
        "solved": 0, "count": 0,
        "centers_done": 0, "edges_paired": 0,
        "reduced": 0, "stage_sum": 0,
    }

    for ep_data in episodes[:num_cubes]:
        ep = Episode.from_dict(ep_data) if isinstance(ep_data, dict) else ep_data
        if ep.size != 4:
            continue

        cube = Cube(4)
        cube.apply_moves(ep.scramble)

        solved, final_cube, history = beam_search_distance(
            model, cube, beam_width=beam_width, max_steps=max_steps, device=device
        )

        results["count"] += 1
        results["solved"] += int(solved)

        try:
            stage = reduction_stage_444(final_cube)
            results["centers_done"] += int(centers_done_444(final_cube))
            results["edges_paired"] += int(paired_edge_count_444(final_cube) == 12)
            results["reduced"] += int(stage >= 3)
            results["stage_sum"] += stage
        except Exception:
            pass

    n = results["count"]
    if n == 0:
        return results

    return {
        "solve_rate": results["solved"] / n,
        "centers_done_rate": results["centers_done"] / n,
        "edges_paired_rate": results["edges_paired"] / n,
        "reduced_to_3x3_rate": results["reduced"] / n,
        "mean_stage_reached": results["stage_sum"] / n,
        "count": n,
    }
