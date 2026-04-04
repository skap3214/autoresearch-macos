#!/usr/bin/env python3
"""
Teacher-based 5x5 center solver.

Uses the dwalton 5x5 solver as a teacher to generate training data,
then trains a small MLP policy+value network to solve 5x5 centers.

Usage:
  python solve_5x5_teacher.py --generate --num-cubes 500
  python solve_5x5_teacher.py --train --hours 2
  python solve_5x5_teacher.py --eval
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rubiks import Cube, Move, FACE_ORDER

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLOR_TO_IDX = {c: i for i, c in enumerate(("W", "Y", "G", "B", "R", "O"))}
NUM_COLORS = 6
NUM_CENTER_STICKERS = 54  # 9 per face x 6 faces

# Build the 36-move action space: 18 regular (width=1) + 18 wide (width=2)
ALL_MOVES_5x5: list[Move] = []
for _face in FACE_ORDER:
    for _turns in (1, -1, 2):
        ALL_MOVES_5x5.append(Move(face=_face, depth=1, width=1, turns=_turns))
for _face in FACE_ORDER:
    for _turns in (1, -1, 2):
        ALL_MOVES_5x5.append(Move(face=_face, depth=1, width=2, turns=_turns))

N_ACTIONS = len(ALL_MOVES_5x5)  # 36

MOVE_TO_IDX = {}
for _i, _m in enumerate(ALL_MOVES_5x5):
    MOVE_TO_IDX[(_m.face, _m.width, _m.turns)] = _i

DATA_DIR = _PROJECT_ROOT / "data_5x5_teacher"
DATA_FILE = DATA_DIR / "center_data.pkl"
MODEL_FILE = DATA_DIR / "center_model.pt"

# ---------------------------------------------------------------------------
# 5x5 center detection
# ---------------------------------------------------------------------------

def centers_done_555(cube: Cube) -> bool:
    """Check if all 9 center stickers on each face are uniform (same color).
    For a 5x5, centers are the inner 3x3 grid: rows 1-3, cols 1-3 of face_grid.
    Uses within-face uniformity (not canonical colors) since centers are movable."""
    if cube.size != 5:
        raise ValueError(f"centers_done_555 only valid for 5x5, got {cube.size}")
    for face in FACE_ORDER:
        grid = cube.face_grid(face)
        # Center is the inner 3x3 block: grid[r][c] for r,c in 1..3
        center_color = grid[1][1]
        for r in range(1, 4):
            for c in range(1, 4):
                if grid[r][c] != center_color:
                    return False
    return True


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def extract_center_stickers(cube: Cube) -> list[int]:
    """Extract 54 center sticker color indices from a 5x5 cube.
    For each face, the 9 center stickers are the inner 3x3 grid:
    grid[1][1], grid[1][2], grid[1][3],
    grid[2][1], grid[2][2], grid[2][3],
    grid[3][1], grid[3][2], grid[3][3].
    """
    indices = []
    for face in FACE_ORDER:
        grid = cube.face_grid(face)
        for r in range(1, 4):
            for c in range(1, 4):
                indices.append(COLOR_TO_IDX[grid[r][c]])
    return indices


def move_to_action_idx(move: Move) -> int:
    """Convert a Move to an action index (0-35)."""
    key = (move.face, move.width, move.turns)
    if key not in MOVE_TO_IDX:
        raise ValueError(f"Move {move} not in action space")
    return MOVE_TO_IDX[key]


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def generate_scramble(rng: random.Random, n_moves: int = 30) -> list[Move]:
    """Generate a random scramble of n_moves, mixing regular and wide moves.
    Default 30 moves for 5x5 (more complex than 4x4)."""
    moves = []
    for _ in range(n_moves):
        face = rng.choice(list(FACE_ORDER))
        turns = rng.choice([1, -1, 2])
        width = rng.choice([1, 2])
        moves.append(Move(face=face, depth=1, width=width, turns=turns))
    return moves


def generate_data(num_cubes: int, seed: int = 42) -> dict:
    """Generate training data using dwalton solver as teacher.

    For each scrambled cube:
      1. Solve with dwalton solver
      2. Replay solution, find center-solving prefix
      3. Record (center_stickers, action, distance_remaining) for each step
    """
    from teacher_dwalton import solve_cube_555, _ensure_solver_importable
    _ensure_solver_importable()

    rng = random.Random(seed)
    all_states = []      # list of 54-int lists
    all_actions = []     # list of ints (0-35)
    all_distances = []   # list of ints (distance to centers-done)

    successes = 0
    failures = 0
    already_solved = 0
    t0 = time.time()

    for i in range(num_cubes):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{num_cubes}] successes={successes} failures={failures} "
                  f"already_solved={already_solved} ({rate:.1f} cubes/s)")

        # Create and scramble cube
        cube = Cube(5)
        scramble = generate_scramble(rng)
        cube.apply_moves(scramble)

        # If centers already solved after scramble, skip
        if centers_done_555(cube):
            already_solved += 1
            continue

        # Solve with teacher (with retry logic)
        solution = None
        for attempt in range(3):
            try:
                solution = solve_cube_555(cube.copy())
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  WARNING: Solver failed on cube {i} after 3 attempts: {e}")
                continue

        if solution is None:
            failures += 1
            continue

        # Replay solution to find center-solving prefix
        replay_cube = cube.copy()
        center_prefix_len = None
        for step_idx, move in enumerate(solution):
            replay_cube.apply_move(move)
            if centers_done_555(replay_cube):
                center_prefix_len = step_idx + 1
                break

        if center_prefix_len is None:
            # Centers never became solved (shouldn't happen if full solve works)
            # but handle gracefully
            failures += 1
            continue

        # Now generate training examples for each state in the prefix
        state_cube = cube.copy()
        for step_idx in range(center_prefix_len):
            move = solution[step_idx]
            distance = center_prefix_len - step_idx

            stickers = extract_center_stickers(state_cube)
            try:
                action = move_to_action_idx(move)
            except ValueError:
                # Move not in our action space (e.g. inner slice); skip this cube
                break

            all_states.append(stickers)
            all_actions.append(action)
            all_distances.append(distance)

            state_cube.apply_move(move)

        successes += 1

    elapsed = time.time() - t0
    print(f"\nGeneration complete: {successes} cubes solved, {failures} failures, "
          f"{already_solved} already-solved, {len(all_states)} training examples, "
          f"{elapsed:.1f}s")

    data = {
        "states": all_states,
        "actions": all_actions,
        "distances": all_distances,
        "num_cubes": successes,
    }
    return data


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CenterDataset(Dataset):
    def __init__(self, states, actions, distances):
        self.states = torch.tensor(states, dtype=torch.long)
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.distances = torch.tensor(distances, dtype=torch.float32)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.distances[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CenterPolicyValueNet(nn.Module):
    """Small MLP for 5x5 center solving.

    Input: 54 center sticker color indices -> embed -> flatten -> FC layers
    Output: policy (36-way) + value (scalar distance estimate)
    """

    def __init__(self, embed_dim: int = 32, hidden_dim: int = 512, n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(NUM_COLORS, embed_dim)
        input_dim = NUM_CENTER_STICKERS * embed_dim  # 54 * 32 = 1728

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, N_ACTIONS),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 54) LongTensor of color indices
        Returns:
            policy_logits: (B, 36)
            value: (B,) predicted distance to centers-done
        """
        emb = self.embed(x)           # (B, 54, embed_dim)
        emb = emb.view(emb.size(0), -1)  # (B, 54*embed_dim)
        h = self.trunk(emb)            # (B, hidden_dim)
        policy = self.policy_head(h)   # (B, 36)
        value = self.value_head(h).squeeze(-1)  # (B,)
        return policy, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(hours: float = 2.0, batch_size: int = 256, lr: float = 1e-3,
          policy_weight: float = 1.0, value_weight: float = 0.1):
    """Train the center policy+value net on teacher data."""

    if not DATA_FILE.exists():
        print(f"ERROR: Data file {DATA_FILE} not found. Run --generate first.")
        sys.exit(1)

    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)

    states = data["states"]
    actions = data["actions"]
    distances = data["distances"]
    print(f"Loaded {len(states)} training examples from {data['num_cubes']} cubes")

    # Split into train/val (90/10)
    n = len(states)
    indices = list(range(n))
    random.Random(42).shuffle(indices)
    split = int(0.9 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_states = [states[i] for i in train_idx]
    train_actions = [actions[i] for i in train_idx]
    train_distances = [distances[i] for i in train_idx]

    val_states = [states[i] for i in val_idx]
    val_actions = [actions[i] for i in val_idx]
    val_distances = [distances[i] for i in val_idx]

    train_ds = CenterDataset(train_states, train_actions, train_distances)
    val_ds = CenterDataset(val_states, val_actions, val_distances)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CenterPolicyValueNet().to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(hours * 3600 / 2)  # rough epoch estimate
    )

    best_val_acc = 0.0
    t0 = time.time()
    deadline = t0 + hours * 3600
    epoch = 0

    while time.time() < deadline:
        epoch += 1
        model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct = 0
        total = 0

        for batch_states, batch_actions, batch_distances in train_loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_distances = batch_distances.to(device)

            policy_logits, value_pred = model(batch_states)

            policy_loss = F.cross_entropy(policy_logits, batch_actions)
            value_loss = F.smooth_l1_loss(value_pred, batch_distances)
            loss = policy_weight * policy_loss + value_weight * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * batch_states.size(0)
            total_policy_loss += policy_loss.item() * batch_states.size(0)
            total_value_loss += value_loss.item() * batch_states.size(0)
            preds = policy_logits.argmax(dim=-1)
            correct += (preds == batch_actions).sum().item()
            total += batch_states.size(0)

        scheduler.step()

        avg_loss = total_loss / total if total > 0 else 0
        avg_ploss = total_policy_loss / total if total > 0 else 0
        avg_vloss = total_value_loss / total if total > 0 else 0
        train_acc = correct / total if total > 0 else 0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_vloss = 0.0
        with torch.no_grad():
            for batch_states, batch_actions, batch_distances in val_loader:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                batch_distances = batch_distances.to(device)

                policy_logits, value_pred = model(batch_states)
                preds = policy_logits.argmax(dim=-1)
                val_correct += (preds == batch_actions).sum().item()
                val_total += batch_states.size(0)
                val_vloss += F.smooth_l1_loss(value_pred, batch_distances, reduction='sum').item()

        val_acc = val_correct / val_total if val_total > 0 else 0
        val_vloss_avg = val_vloss / val_total if val_total > 0 else 0
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d} ({elapsed/60:.1f}m) | "
              f"loss={avg_loss:.4f} ploss={avg_ploss:.4f} vloss={avg_vloss:.4f} | "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} val_vloss={val_vloss_avg:.3f} | "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_vloss": val_vloss_avg,
            }, MODEL_FILE)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete after {epoch} epochs. Best val_acc={best_val_acc:.4f}")


# ---------------------------------------------------------------------------
# Solving / Evaluation
# ---------------------------------------------------------------------------

def solve_greedy(cube: Cube, model: CenterPolicyValueNet, device: torch.device,
                 max_steps: int = 80) -> tuple[bool, list[Move], int]:
    """Solve centers by greedily following the policy's top prediction."""
    current = cube.copy()
    moves = []
    for step in range(max_steps):
        if centers_done_555(current):
            return True, moves, step
        stickers = extract_center_stickers(current)
        x = torch.tensor([stickers], dtype=torch.long, device=device)
        with torch.no_grad():
            policy_logits, _ = model(x)
        action = policy_logits.argmax(dim=-1).item()
        move = ALL_MOVES_5x5[action]
        current.apply_move(move)
        moves.append(move)
    return centers_done_555(current), moves, max_steps


def solve_beam(cube: Cube, model: CenterPolicyValueNet, device: torch.device,
               beam_width: int = 32, max_steps: int = 80, top_k: int = 5
               ) -> tuple[bool, list[Move], int]:
    """Solve centers using policy-guided beam search with value ranking."""
    if centers_done_555(cube):
        return True, [], 0

    @dataclass
    class BeamItem:
        cube: Cube
        moves: list
        score: float  # negative value (lower = better)

    beam = [BeamItem(cube=cube.copy(), moves=[], score=0.0)]
    visited = {cube.to_kociemba_string()}
    nodes = 0

    for step in range(max_steps):
        if not beam:
            break

        candidates = []
        # Batch encode all beam states
        all_stickers = []
        for item in beam:
            all_stickers.append(extract_center_stickers(item.cube))

        x = torch.tensor(all_stickers, dtype=torch.long, device=device)
        with torch.no_grad():
            policy_logits, values = model(x)
            probs = F.softmax(policy_logits, dim=-1)

        for i, item in enumerate(beam):
            top_actions = probs[i].topk(top_k).indices.tolist()
            for action in top_actions:
                move = ALL_MOVES_5x5[action]
                child = item.cube.copy()
                child.apply_move(move)
                nodes += 1

                if centers_done_555(child):
                    return True, item.moves + [move], nodes

                key = child.to_kociemba_string()
                if key in visited:
                    continue
                visited.add(key)

                # Score child by value prediction
                child_stickers = extract_center_stickers(child)
                cx = torch.tensor([child_stickers], dtype=torch.long, device=device)
                with torch.no_grad():
                    _, child_val = model(cx)
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


def evaluate(num_cubes: int = 200, scramble_len: int = 30, seed: int = 123):
    """Evaluate the trained model on random scrambles."""
    if not MODEL_FILE.exists():
        print(f"ERROR: Model file {MODEL_FILE} not found. Run --train first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CenterPolicyValueNet().to(device)
    ckpt = torch.load(MODEL_FILE, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")
    print(f"Model parameters: {model.count_parameters():,}")

    rng = random.Random(seed)

    # Test greedy solving
    print(f"\n--- Greedy Evaluation ({num_cubes} cubes, {scramble_len}-move scrambles) ---")
    greedy_solved = 0
    greedy_lengths = []
    for i in range(num_cubes):
        cube = Cube(5)
        scramble = generate_scramble(rng, scramble_len)
        cube.apply_moves(scramble)

        if centers_done_555(cube):
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

    # Test beam search
    print(f"\n--- Beam Search Evaluation ({min(num_cubes, 50)} cubes, beam=32) ---")
    rng2 = random.Random(seed)  # same scrambles
    beam_solved = 0
    beam_lengths = []
    beam_n = min(num_cubes, 50)  # beam is slower, test fewer
    for i in range(beam_n):
        cube = Cube(5)
        scramble = generate_scramble(rng2, scramble_len)
        cube.apply_moves(scramble)

        if centers_done_555(cube):
            beam_solved += 1
            beam_lengths.append(0)
            continue

        solved, moves, nodes = solve_beam(cube, model, device, beam_width=32)
        if solved:
            beam_solved += 1
            beam_lengths.append(len(moves))

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{beam_n}] solved={beam_solved}")

    print(f"Beam solve rate: {beam_solved}/{beam_n} = {beam_solved/beam_n:.1%}")
    if beam_lengths:
        print(f"  Avg solution length: {sum(beam_lengths)/len(beam_lengths):.1f} moves")
        print(f"  Max solution length: {max(beam_lengths)} moves")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="5x5 center solver via teacher distillation")
    parser.add_argument("--generate", action="store_true", help="Generate training data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--num-cubes", type=int, default=500, help="Number of cubes for data generation")
    parser.add_argument("--hours", type=float, default=2.0, help="Training time in hours")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not any([args.generate, args.train, args.eval]):
        parser.print_help()
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.generate:
        print(f"Generating data from {args.num_cubes} cubes...")
        data = generate_data(args.num_cubes, seed=args.seed)
        with open(DATA_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved to {DATA_FILE}")
        print(f"  Total examples: {len(data['states'])}")
        if data['distances']:
            avg_dist = sum(data['distances']) / len(data['distances'])
            max_dist = max(data['distances'])
            print(f"  Avg distance: {avg_dist:.1f}, Max distance: {max_dist}")

    if args.train:
        train(hours=args.hours)

    if args.eval:
        evaluate()


if __name__ == "__main__":
    main()
