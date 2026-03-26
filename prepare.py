"""
One-time data preparation for Rubik's-cube autoresearch experiments.

This version replaces the generic text corpus with a synthetic cube-policy
dataset built from canonicalized NxN cube states and structured move tokens.

Usage:
    python prepare.py
    python prepare.py --force

Artifacts are stored in ~/.cache/autoresearch-rubiks/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
from collections import defaultdict
from contextlib import nullcontext

import torch

from rubiks import (
    Cube,
    Episode,
    Move,
    build_answer_tokens,
    build_prompt_tokens,
    build_training_examples_from_solution,
    parse_answer_tokens,
    random_scramble,
    scramble_length_for_size,
    get_symmetry_rotations,
    transform_episode,
)
from teacher_dwalton import solve_cube_222, solve_cube_333
# Lazy imports — not used in data generation, avoid import-time crashes
# from teacher_cfop import solve_cube_333_cfop
# from teacher_pycuber import solve_cube_333_pycuber

# ---------------------------------------------------------------------------
# Constants (fixed for v1)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 74
TIME_BUDGET = 43200  # 12 hours — generous with 24x symmetry augmented data
TEACHER_BACKEND = "dwalton76/rubiks-cube-NxNxN-solver"
PROMPT_FORMAT_VERSION = "flat24-history3-jointmove-kociemba-v2"

TRAIN_SIZES = (2, 3)
ID_VAL_SIZES = TRAIN_SIZES
OOD_DEV_SIZES = ()
OOD_TEST_SIZES = ()

TRAIN_EPISODES_PER_SIZE = 65536  # base
_TRAIN_EPISODES_OVERRIDE = {3: 131072}  # 2x base; 24x symmetry aug gives 94M effective examples
ID_VAL_EPISODES_PER_SIZE = 256
OOD_DEV_EPISODES_PER_SIZE = 0
OOD_TEST_EPISODES_PER_SIZE = 0

MAX_MOVE_DEPTH = 2
MAX_MOVE_WIDTH = 2
MAX_GENERATION_TOKENS = 20

TRAIN_RNG_SEED = 42
VAL_RNG_SEED = 99
TRAIN_USE_CURRICULUM = False
TRAIN_CURRICULUM_SCRAMBLE_LENGTHS = (2, 4, 6, 8, 10, 14, 18)

# Symmetry augmentation: apply N of the 24 cube rotational symmetries to each episode.
# 0 = no augmentation (just original), 24 = full augmentation (24x data).
# Each rotation produces a valid but distinct training example from the same episode.
SYMMETRY_AUGMENTATION_COUNT = 1  # no pre-computed aug; online augmentation in dataloader instead

ROLLOUT_MIN_STEPS = 200
SEARCH_SELECTOR = "hybrid_greedy_v1"  # fast eval; value-guided tested post-hoc
SEARCH_RESIDUAL_DELTA = 2
SEARCH_LOOKAHEAD_TOP_K = 3
SEARCH_SECOND_SCORE_DISCOUNT = 0.5


def _stable_version(prefix: str, payload: dict[str, object]) -> str:
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


DATA_CONFIG = {
    "prompt_format_version": PROMPT_FORMAT_VERSION,
    "teacher_backend": TEACHER_BACKEND,
    "train_sizes": TRAIN_SIZES,
    "id_val_sizes": ID_VAL_SIZES,
    "ood_dev_sizes": OOD_DEV_SIZES,
    "ood_test_sizes": OOD_TEST_SIZES,
    "train_episodes_per_size": TRAIN_EPISODES_PER_SIZE,
    "id_val_episodes_per_size": ID_VAL_EPISODES_PER_SIZE,
    "ood_dev_episodes_per_size": OOD_DEV_EPISODES_PER_SIZE,
    "ood_test_episodes_per_size": OOD_TEST_EPISODES_PER_SIZE,
    "max_move_depth": MAX_MOVE_DEPTH,
    "max_move_width": MAX_MOVE_WIDTH,
    "train_rng_seed": TRAIN_RNG_SEED,
    "val_rng_seed": VAL_RNG_SEED,
    "train_use_curriculum": TRAIN_USE_CURRICULUM,
    "train_curriculum_scramble_lengths": TRAIN_CURRICULUM_SCRAMBLE_LENGTHS,
    "symmetry_augmentation_count": SYMMETRY_AUGMENTATION_COUNT,
}
DATA_VERSION = _stable_version("rubiks-v9", DATA_CONFIG)


def get_experiment_manifest() -> dict[str, object]:
    return {
        "data": {
            "data_version": DATA_VERSION,
            "dataset_path": DATASET_PATH,
            **DATA_CONFIG,
        },
        "eval": {
            "rollout_min_steps": ROLLOUT_MIN_STEPS,
            "search_selector": SEARCH_SELECTOR,
            "search_residual_delta": SEARCH_RESIDUAL_DELTA,
            "search_lookahead_top_k": SEARCH_LOOKAHEAD_TOP_K,
            "search_second_score_discount": SEARCH_SECOND_SCORE_DISCOUNT,
            "no_inverse_rule": True,
            "state_avoidance": True,
        },
    }

# ---------------------------------------------------------------------------
# Artifact locations
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-rubiks")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
DATASET_PATH = os.path.join(DATA_DIR, "datasets.pkl")
TOKENIZER_PATH = os.path.join(TOKENIZER_DIR, "tokenizer.json")

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def report_environment():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Environment detected: {device}")
    print()


def get_runtime_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def build_vocab() -> list[str]:
    tokens = [
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<TASK_POLICY>",
        "<SIZE>",
        "</SIZE>",
        "<STATE>",
        "</STATE>",
        "<TARGET>",
        "<DONE>",
        "<MOVE>",
        "</MOVE>",
        "<FACE>",
        "</FACE>",
        "<DEPTH>",
        "</DEPTH>",
        "<WIDTH>",
        "</WIDTH>",
        "<TURN>",
        "</TURN>",
        "<ROW>",
        "</ROW>",
    ]
    for face in ("U", "R", "F", "D", "L", "B"):
        tokens.append(f"<GRID_{face}>")
        tokens.append(f"</GRID_{face}>")
        tokens.append(f"FACE_{face}")
    for color in ("W", "Y", "G", "B", "R", "O"):
        tokens.append(f"COL_{color}")
    for turn in ("CW", "CCW", "HALF"):
        tokens.append(f"TURN_{turn}")
    for digit in range(10):
        tokens.append(f"DIGIT_{digit}")
    # Joint move tokens: MOVE_{face}_{turn} for single-token move prediction
    for face in ("U", "R", "F", "D", "L", "B"):
        for turn in ("CW", "CCW", "HALF"):
            tokens.append(f"MOVE_{face}_{turn}")
    # CFOP stage-conditioning tokens
    tokens.append("STAGE_CROSS")
    tokens.append("STAGE_F2L")
    tokens.append("STAGE_OLL")
    tokens.append("STAGE_PLL")
    tokens.append("STAGE_SOLVED")
    return tokens


class Tokenizer:
    def __init__(self, token_to_id: dict[str, int], id_to_token: list[str]):
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR) -> "Tokenizer":
        with open(os.path.join(tokenizer_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(payload["token_to_id"], payload["id_to_token"])

    def save(self, tokenizer_dir=TOKENIZER_DIR):
        os.makedirs(tokenizer_dir, exist_ok=True)
        payload = {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
        }
        with open(os.path.join(tokenizer_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def get_vocab_size(self) -> int:
        return len(self.id_to_token)

    def get_pad_token_id(self) -> int:
        return self.token_to_id["<|pad|>"]

    def get_bos_token_id(self) -> int:
        return self.token_to_id["<|bos|>"]

    def get_eos_token_id(self) -> int:
        return self.token_to_id["<|eos|>"]

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        try:
            return [self.token_to_id[token] for token in tokens]
        except KeyError as exc:
            raise ValueError(f"Unknown token: {exc.args[0]}") from exc

    def decode_ids(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[idx] for idx in ids]

    def encode(self, text, prepend=None, num_threads=8):
        del num_threads
        if isinstance(text, str):
            tokens = text.strip().split()
            ids = self.encode_tokens(tokens)
            if prepend is not None:
                prepend_id = prepend if isinstance(prepend, int) else self.token_to_id[prepend]
                ids.insert(0, prepend_id)
            return ids
        if isinstance(text, list):
            encoded = []
            for item in text:
                ids = self.encode(item, prepend=prepend)
                encoded.append(ids)
            return encoded
        raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.decode_ids(ids))


def ensure_tokenizer(force: bool = False):
    if not force and os.path.exists(TOKENIZER_PATH):
        print(f"Tokenizer: already present at {TOKENIZER_DIR}")
        return

    tokenizer = Tokenizer(
        token_to_id={token: idx for idx, token in enumerate(build_vocab())},
        id_to_token=build_vocab(),
    )
    tokenizer.save(TOKENIZER_DIR)
    print(f"Tokenizer: wrote fixed vocabulary with {tokenizer.get_vocab_size()} tokens")


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def encode_supervised_example(tokenizer: Tokenizer, prompt_tokens: list[str], answer_tokens: list[str]) -> dict[str, object]:
    prompt_ids = tokenizer.encode_tokens(prompt_tokens)
    answer_ids = tokenizer.encode_tokens(answer_tokens)
    full_ids = [tokenizer.get_bos_token_id(), *prompt_ids, *answer_ids]
    input_ids = full_ids[:-1]
    targets = [-1] * len(prompt_ids) + full_ids[len(prompt_ids) + 1:]
    if len(input_ids) != len(targets):
        raise AssertionError("input/target length mismatch")
    if len(input_ids) > MAX_SEQ_LEN:
        raise ValueError(f"Example length {len(input_ids)} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}")
    return {
        "input_ids": input_ids,
        "targets": targets,
        "prompt_len": len(prompt_ids),
        "answer_len": len(answer_ids),
    }


def episode_to_examples(tokenizer: Tokenizer, episode: Episode) -> list[dict[str, object]]:
    # Use sub-goal training for CFOP episodes (3x3 with staged solutions)
    staged = getattr(episode, '_staged_solution', None)
    if staged:
        from rubiks import build_subgoal_training_examples
        examples = []
        for prompt_tokens, answer_tokens, distance in build_subgoal_training_examples(
            episode.size, episode.scramble, staged,
        ):
            try:
                encoded = encode_supervised_example(tokenizer, prompt_tokens, answer_tokens)
                encoded["size"] = episode.size
                encoded["distance_to_goal"] = distance
                examples.append(encoded)
            except (ValueError, AssertionError):
                continue  # skip if sequence too long
        return examples

    examples = []
    for prompt_tokens, answer_tokens, distance in build_training_examples_from_solution(
        episode.size,
        episode.scramble,
        episode.solution,
    ):
        encoded = encode_supervised_example(tokenizer, prompt_tokens, answer_tokens)
        encoded["size"] = episode.size
        encoded["distance_to_goal"] = distance
        examples.append(encoded)
    return examples


def _augment_one_worker(args):
    """Module-level worker for parallel symmetry augmentation + tokenization."""
    import sys
    solver_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rubiks-cube-NxNxN-solver")
    if os.path.exists(solver_root) and solver_root not in sys.path:
        sys.path.insert(0, solver_root)
    episode_dict, rot_idx = args
    episode = Episode.from_dict(episode_dict)
    aug_episode = transform_episode(episode, rot_idx)
    tokenizer = Tokenizer(
        token_to_id={token: idx for idx, token in enumerate(build_vocab())},
        id_to_token=build_vocab(),
    )
    return episode_to_examples(tokenizer, aug_episode)


def _generate_one_worker(args):
    """Module-level worker for parallel episode generation."""
    import sys
    solver_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rubiks-cube-NxNxN-solver")
    if os.path.exists(solver_root) and solver_root not in sys.path:
        sys.path.insert(0, solver_root)
    size, seed, scramble_length = args
    worker_rng = random.Random(seed)
    return generate_teacher_episode(size=size, rng=worker_rng, scramble_length=scramble_length)


def generate_teacher_episode(size: int, rng: random.Random, scramble_length: int | None = None) -> Episode:
    if scramble_length is None:
        scramble_length = scramble_length_for_size(size)
    scramble = random_scramble(
        size=size,
        length=scramble_length,
        rng=rng,
        max_depth=MAX_MOVE_DEPTH,
        max_width=MAX_MOVE_WIDTH,
    )
    cube = Cube(size)
    cube.apply_moves(scramble)

    if size == 2:
        solution = solve_cube_222(cube)
        return Episode(
            size=size,
            scramble=scramble,
            solution=solution,
            max_rollout_steps=max(8, len(solution) * 2),
        )
    elif size == 3:
        # Kociemba teacher: short ~20-move solutions, best for beam search
        solution = solve_cube_333(cube)
        return Episode(
            size=size,
            scramble=scramble,
            solution=solution,
            max_rollout_steps=max(8, len(solution) * 2),
        )
    else:
        raise NotImplementedError(f"Teacher backend is not integrated for size {size} yet")


def _generate_perturbed_examples(
    tokenizer: Tokenizer, episode: Episode, rng: random.Random, num_perturbations: int = 2,
) -> list[dict[str, object]]:
    """DAgger-style data augmentation: inject random wrong moves into teacher
    solution paths, then query the teacher for corrections from the resulting
    off-policy states. This teaches the model to recover from mistakes."""
    if len(episode.solution) == 0:
        return []
    examples = []
    for _ in range(num_perturbations):
        cube = Cube(episode.size)
        cube.apply_moves(episode.scramble)
        history: list[Move] = []

        # Follow teacher solution for a random prefix
        inject_point = rng.randint(0, len(episode.solution) - 1)
        for i in range(inject_point):
            cube.apply_move(episode.solution[i])
            history.append(episode.solution[i])

        # Inject 1-2 random moves (creating an off-policy state)
        num_random = rng.randint(1, 2)
        for _ in range(num_random):
            face = rng.choice(("U", "R", "F", "D", "L", "B"))
            turns = rng.choice((1, -1, 2))
            wrong_move = Move(face=face, depth=1, width=1, turns=turns)
            cube.apply_move(wrong_move)
            history.append(wrong_move)

        if cube.has_uniform_faces():
            continue  # already solved by chance, skip

        # Query teacher for correction from this perturbed state
        try:
            correction = solve_cube_222(cube)
        except Exception:
            continue

        if not correction:
            continue

        # Add the first correction step as a training example
        prompt_tokens = build_prompt_tokens(episode.size, cube, history=history)
        answer_tokens = build_answer_tokens(correction[0])
        try:
            encoded = encode_supervised_example(tokenizer, prompt_tokens, answer_tokens)
            encoded["size"] = episode.size
            examples.append(encoded)
        except (ValueError, AssertionError):
            continue

        # Optionally add a few more steps from the correction path
        for step_idx in range(1, min(3, len(correction))):
            cube.apply_move(correction[step_idx - 1])
            history.append(correction[step_idx - 1])
            if cube.has_uniform_faces():
                break
            prompt_tokens = build_prompt_tokens(episode.size, cube, history=history)
            answer_tokens = build_answer_tokens(correction[step_idx])
            try:
                encoded = encode_supervised_example(tokenizer, prompt_tokens, answer_tokens)
                encoded["size"] = episode.size
                examples.append(encoded)
            except (ValueError, AssertionError):
                break

    return examples


def build_dataset_payload(force: bool = False):
    if not force and os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "rb") as f:
            payload = pickle.load(f)
        if payload.get("data_version") == DATA_VERSION:
            print(f"Data: already prepared at {DATASET_PATH}")
            return

    os.makedirs(DATA_DIR, exist_ok=True)
    tokenizer = Tokenizer.from_directory()
    train_rng = random.Random(TRAIN_RNG_SEED)
    val_rng = random.Random(VAL_RNG_SEED)

    train_examples: list[dict[str, object]] = []
    id_val_examples: list[dict[str, object]] = []
    ood_dev_examples: list[dict[str, object]] = []
    eval_episodes = {
        "id": [],
        "ood_dev": [],
        "ood_test": [],
    }

    def generate_episode_batch(size: int, count: int, rng: random.Random, curriculum: bool = False) -> list[Episode]:
        from multiprocessing import Pool, cpu_count
        args = []
        for _ in range(count):
            sl = rng.choice(TRAIN_CURRICULUM_SCRAMBLE_LENGTHS) if curriculum else None
            seed = rng.randint(0, 2**31)
            args.append((size, seed, sl))
        n_workers = min(cpu_count(), 16)
        print(f"  Generating {count} episodes for size {size} with {n_workers} workers...")
        with Pool(n_workers) as pool:
            episodes = pool.map(_generate_one_worker, args, chunksize=max(1, count // (n_workers * 4)))
        return episodes

    # Determine which symmetry rotation indices to use
    n_sym = min(SYMMETRY_AUGMENTATION_COUNT, 24)
    sym_indices = list(range(n_sym))  # 0 = identity, 1-23 = rotations
    if n_sym > 0:
        print(f"  Symmetry augmentation: {n_sym}x (using {n_sym} of 24 rotations)")

    for size in TRAIN_SIZES:
        n_episodes = _TRAIN_EPISODES_OVERRIDE.get(size, TRAIN_EPISODES_PER_SIZE)
        episodes = generate_episode_batch(
            size,
            n_episodes,
            rng=train_rng,
            curriculum=TRAIN_USE_CURRICULUM,
        )
        if n_sym <= 1:
            # No augmentation — sequential (fast enough)
            for episode in episodes:
                train_examples.extend(episode_to_examples(tokenizer, episode))
        else:
            # Parallel augmentation + tokenization
            from multiprocessing import Pool, cpu_count
            aug_args = [
                (episode.to_dict(), rot_idx)
                for episode in episodes
                for rot_idx in sym_indices
            ]
            n_workers = min(cpu_count(), 16)
            total = len(aug_args)
            print(f"  Augmenting {len(episodes)} episodes x {n_sym} rotations = {total} tasks with {n_workers} workers...")
            with Pool(n_workers) as pool:
                for i, examples in enumerate(pool.imap_unordered(_augment_one_worker, aug_args, chunksize=256)):
                    train_examples.extend(examples)
                    if (i + 1) % 50000 == 0:
                        print(f"    {i+1}/{total} augmented ({len(train_examples):,} examples so far)")
            print(f"    Done: {len(train_examples):,} train examples for size {size}")

    for size in ID_VAL_SIZES:
        episodes = generate_episode_batch(size, ID_VAL_EPISODES_PER_SIZE, rng=val_rng)
        eval_episodes["id"].extend(episode.to_dict() for episode in episodes)
        for episode in episodes:
            id_val_examples.extend(episode_to_examples(tokenizer, episode))

    for size in OOD_DEV_SIZES:
        episodes = generate_episode_batch(size, OOD_DEV_EPISODES_PER_SIZE, rng=val_rng)
        eval_episodes["ood_dev"].extend(episode.to_dict() for episode in episodes)
        for episode in episodes:
            ood_dev_examples.extend(episode_to_examples(tokenizer, episode))

    for size in OOD_TEST_SIZES:
        episodes = generate_episode_batch(size, OOD_TEST_EPISODES_PER_SIZE, rng=val_rng)
        eval_episodes["ood_test"].extend(episode.to_dict() for episode in episodes)

    payload = {
        "data_version": DATA_VERSION,
        "config": {
            "max_seq_len": MAX_SEQ_LEN,
            **DATA_CONFIG,
        },
        "train_examples": train_examples,
        "id_val_examples": id_val_examples,
        "ood_dev_examples": ood_dev_examples,
        "eval_episodes": eval_episodes,
    }
    with open(DATASET_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(
        "Data: wrote "
        f"{len(train_examples):,} train examples, "
        f"{len(id_val_examples):,} ID val examples, "
        f"{len(ood_dev_examples):,} OOD-dev val examples"
    )
    print(f"Data: evaluation suite stored at {DATASET_PATH}")


def load_dataset() -> dict[str, object]:
    with open(DATASET_PATH, "rb") as f:
        payload = pickle.load(f)
    if payload.get("data_version") != DATA_VERSION:
        raise RuntimeError("Dataset version mismatch. Re-run prepare.py --force")
    return payload


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------


def _example_stream(examples: list[dict[str, object]], shuffle: bool):
    order = list(range(len(examples)))
    rng = random.Random(1234 if shuffle else 0)
    epoch = 1
    while True:
        if shuffle:
            rng.shuffle(order)
        for idx in order:
            yield examples[idx], epoch
        epoch += 1


# ---------------------------------------------------------------------------
# Online symmetry augmentation — precomputed tables
# ---------------------------------------------------------------------------

_STICKER_PERMS: dict[int, list[list[int]]] | None = None
_MOVE_TOKEN_MAP: list[dict[int, int]] | None = None


def _build_sticker_permutation(size: int, rot_idx: int) -> list[int]:
    """Compute how flat sticker indices permute under rotation rot_idx.
    Returns perm where: rotated_stickers[i] = original_stickers[perm[i]]."""
    from rubiks import Cube, _apply_whole_cube_rotation, get_symmetry_rotations, FACE_ORDER
    rotations = get_symmetry_rotations()
    rot = rotations[rot_idx]
    n_stickers = 6 * size * size

    # Create a cube with unique "colors" at each position to track permutation
    solved = Cube(size)
    # Assign unique IDs via sticker dict — read flat order to get original indices
    orig_flat = []
    for face in FACE_ORDER:
        for row in solved.face_grid(face):
            orig_flat.extend(row)

    # Apply rotation
    rotated = solved.copy()
    for axis, qt in rot:
        rotated = _apply_whole_cube_rotation(rotated, axis, qt)

    rot_flat = []
    for face in FACE_ORDER:
        for row in rotated.face_grid(face):
            rot_flat.extend(row)

    # Build permutation: for each position in rotated, find where that color
    # came from in the original. Since solved has fixed colors per face, we need
    # a different approach — use position tracking instead.

    # Better: create cube with unique marker per sticker position
    # We'll use the sticker (position, normal) keys directly
    solved2 = Cube(size)
    # Map each sticker to its flat index
    sticker_to_idx = {}
    idx = 0
    for face in FACE_ORDER:
        normal = solved2._get_face_normal(face) if hasattr(solved2, '_get_face_normal') else None
        from rubiks import FACE_NORMALS, FACE_VIEW_BASIS, _dot
        normal = FACE_NORMALS[face]
        up_vec, right_vec = FACE_VIEW_BASIS[face]
        for r in range(size):
            for c in range(size):
                sticker_to_idx[(face, r, c)] = idx
                idx += 1

    # For the rotated cube, figure out which original (face, r, c) maps to each
    # new (face, r, c) position
    from rubiks import rotate_vec
    perm = [0] * n_stickers
    for (position, normal_vec), color in solved2.stickers.items():
        # Rotate this sticker
        new_pos = position
        new_norm = normal_vec
        for axis, qt in rot:
            new_pos = rotate_vec(new_pos, axis, qt)
            new_norm = rotate_vec(new_norm, axis, qt)

        # Find which face the rotated normal belongs to
        new_face = None
        for f, fn in FACE_NORMALS.items():
            if fn == new_norm:
                new_face = f
                break

        # Find (row, col) in the new face
        up_vec, right_vec = FACE_VIEW_BASIS[new_face]
        row_val = -_dot(new_pos, up_vec)
        col_val = _dot(new_pos, right_vec)
        new_r = (row_val + (size - 1)) // 2
        new_c = (col_val + (size - 1)) // 2

        # Find original (face, r, c)
        orig_face = None
        for f, fn in FACE_NORMALS.items():
            if fn == normal_vec:
                orig_face = f
                break
        orig_up, orig_right = FACE_VIEW_BASIS[orig_face]
        orig_row_val = -_dot(position, orig_up)
        orig_col_val = _dot(position, orig_right)
        orig_r = (orig_row_val + (size - 1)) // 2
        orig_c = (orig_col_val + (size - 1)) // 2

        orig_idx = sticker_to_idx[(orig_face, orig_r, orig_c)]
        new_idx = sticker_to_idx[(new_face, new_r, new_c)]
        perm[new_idx] = orig_idx

    return perm


def _build_move_token_map(tokenizer: Tokenizer) -> list[dict[int, int]]:
    """For each of 24 rotations, map move token IDs to transformed token IDs."""
    from rubiks import get_move_transform_table, FACE_ORDER
    table = get_move_transform_table()
    turn_names = {1: "CW", -1: "CCW", 2: "HALF"}
    maps = []
    for rot_idx in range(24):
        token_map = {}
        for face in FACE_ORDER:
            for turns, tname in turn_names.items():
                orig_token = f"MOVE_{face}_{tname}"
                new_face, new_turns = table[rot_idx][(face, turns)]
                new_token = f"MOVE_{new_face}_{turn_names[new_turns]}"
                orig_id = tokenizer.token_to_id[orig_token]
                new_id = tokenizer.token_to_id[new_token]
                token_map[orig_id] = new_id
        maps.append(token_map)
    return maps


_COLOR_TOKEN_MAP: list[dict[int, int]] | None = None


def _build_color_token_map(tokenizer: Tokenizer) -> list[dict[int, int]]:
    """For each of 24 rotations, map color token IDs to relabeled color token IDs.
    Under rotation R, face X maps to face Y, so COL_{color_of_X} → COL_{color_of_Y}."""
    from rubiks import get_symmetry_rotations, FACE_ORDER, FACE_NORMALS, FACE_COLORS, rotate_vec
    rotations = get_symmetry_rotations()
    normal_to_face = {v: k for k, v in FACE_NORMALS.items()}
    maps = []
    for rot_idx in range(24):
        rot = rotations[rot_idx]
        token_map = {}
        for face in FACE_ORDER:
            normal = FACE_NORMALS[face]
            for axis, qt in rot:
                normal = rotate_vec(normal, axis, qt)
            new_face = normal_to_face[normal]
            orig_color = FACE_COLORS[face]
            new_color = FACE_COLORS[new_face]
            orig_id = tokenizer.token_to_id[f"COL_{orig_color}"]
            new_id = tokenizer.token_to_id[f"COL_{new_color}"]
            token_map[orig_id] = new_id
        maps.append(token_map)
    return maps


def _get_augmentation_tables(tokenizer: Tokenizer):
    """Lazy-init and return (sticker_perms, move_token_maps, color_token_maps)."""
    global _STICKER_PERMS, _MOVE_TOKEN_MAP, _COLOR_TOKEN_MAP
    if _STICKER_PERMS is None:
        _STICKER_PERMS = {}
        for size in (2, 3):
            _STICKER_PERMS[size] = [_build_sticker_permutation(size, r) for r in range(24)]
        _MOVE_TOKEN_MAP = _build_move_token_map(tokenizer)
        _COLOR_TOKEN_MAP = _build_color_token_map(tokenizer)
    return _STICKER_PERMS, _MOVE_TOKEN_MAP, _COLOR_TOKEN_MAP


def _augment_example_online(input_ids: list[int], target_ids: list[int],
                            size: int, rot_idx: int,
                            sticker_perms: dict, move_maps: list,
                            color_maps: list) -> tuple[list[int], list[int]]:
    """Apply rotation rot_idx to a single tokenized example.

    Three transformations:
    1. Permute sticker positions (which physical position maps where)
    2. Relabel sticker colors (W→G if U face maps to F face)
    3. Transform move tokens (face label mapping)
    """
    if rot_idx == 0:
        return input_ids, target_ids

    n_stickers = 6 * size * size
    perm = sticker_perms[size][rot_idx]
    move_map = move_maps[rot_idx]
    color_map = color_maps[rot_idx]

    # Find sticker start position
    # 2x2: [BOS, TASK_POLICY, SIZE, DIGIT_N, /SIZE, stickers...]  → start=5
    # 3x3: [BOS, TASK_POLICY, SIZE, DIGIT_N, /SIZE, STAGE_X, stickers...] → start=6
    sticker_start = 6 if size >= 3 else 5
    sticker_end = sticker_start + n_stickers

    new_input = list(input_ids)
    new_target = list(target_ids)

    # 1. Permute sticker positions, 2. Relabel colors
    if sticker_end <= len(new_input):
        orig_stickers = input_ids[sticker_start:sticker_end]
        for i, src in enumerate(perm):
            token_id = orig_stickers[src]
            # Relabel color after permuting position
            new_input[sticker_start + i] = color_map.get(token_id, token_id)

    # 3. Transform move tokens in input (history moves after stickers)
    for i in range(sticker_end, len(new_input)):
        if new_input[i] in move_map:
            new_input[i] = move_map[new_input[i]]

    # 3. Transform answer token in targets
    for i in range(len(new_target)):
        if new_target[i] in move_map:
            new_target[i] = move_map[new_target[i]]

    return new_input, new_target


def make_dataloader(tokenizer: Tokenizer, B: int, T: int, split: str):
    if split not in {"train", "val", "ood_val"}:
        raise ValueError(f"Unsupported split: {split}")
    payload = load_dataset()
    if split == "train":
        examples = payload["train_examples"]
        shuffle = True
    elif split == "val":
        examples = payload["id_val_examples"]
        shuffle = False
    else:
        examples = payload["ood_dev_examples"]
        shuffle = False

    # Online symmetry augmentation (train split only)
    use_augmentation = (split == "train") and SYMMETRY_AUGMENTATION_COUNT <= 1
    sticker_perms = move_maps = color_maps = None
    if use_augmentation:
        sticker_perms, move_maps, color_maps = _get_augmentation_tables(tokenizer)
        print(f"  Online symmetry augmentation: enabled (24 rotations per example)")
    aug_rng = random.Random(12345)

    stream = _example_stream(examples, shuffle=shuffle)
    device = get_runtime_device()
    pad_id = tokenizer.get_pad_token_id()

    cpu_inputs = torch.full((B, T), pad_id, dtype=torch.long, pin_memory=(device == "cuda"))
    cpu_targets = torch.full((B, T), -1, dtype=torch.long, pin_memory=(device == "cuda"))
    cpu_distances = torch.zeros(B, dtype=torch.float32, pin_memory=(device == "cuda"))
    inputs = torch.full((B, T), pad_id, dtype=torch.long, device=device)
    targets = torch.full((B, T), -1, dtype=torch.long, device=device)
    distances = torch.zeros(B, dtype=torch.float32, device=device)

    while True:
        for row_idx in range(B):
            example, epoch = next(stream)
            input_ids = example["input_ids"]
            target_ids = example["targets"]

            # Apply random rotation (online augmentation)
            if use_augmentation:
                size = example.get("size", 2)
                rot_idx = aug_rng.randint(0, 23)
                if size in sticker_perms:
                    input_ids, target_ids = _augment_example_online(
                        input_ids, target_ids, size, rot_idx,
                        sticker_perms, move_maps, color_maps
                    )

            seq_len = min(len(input_ids), T)
            cpu_inputs[row_idx].fill_(pad_id)
            cpu_targets[row_idx].fill_(-1)
            cpu_inputs[row_idx, :seq_len] = torch.tensor(input_ids[:seq_len], dtype=torch.long)
            cpu_targets[row_idx, :seq_len] = torch.tensor(target_ids[:seq_len], dtype=torch.long)
            cpu_distances[row_idx] = float(example.get("distance_to_goal", 0))

        inputs.copy_(cpu_inputs, non_blocking=(device == "cuda"))
        targets.copy_(cpu_targets, non_blocking=(device == "cuda"))
        distances.copy_(cpu_distances, non_blocking=(device == "cuda"))
        yield inputs, targets, distances, epoch


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _autocast_context(device_type: str):
    if device_type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device_type == "cpu":
        return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def _tensorize_batch(examples: list[dict[str, object]], tokenizer: Tokenizer, T: int, device: str):
    pad_id = tokenizer.get_pad_token_id()
    inputs = torch.full((len(examples), T), pad_id, dtype=torch.long, device=device)
    targets = torch.full((len(examples), T), -1, dtype=torch.long, device=device)
    for row_idx, example in enumerate(examples):
        seq_len = min(len(example["input_ids"]), T)
        inputs[row_idx, :seq_len] = torch.tensor(example["input_ids"][:seq_len], dtype=torch.long, device=device)
        targets[row_idx, :seq_len] = torch.tensor(example["targets"][:seq_len], dtype=torch.long, device=device)
    return inputs, targets


@torch.no_grad()
def evaluate_move_accuracy(model, tokenizer: Tokenizer, examples: list[dict[str, object]], batch_size: int) -> float:
    if not examples:
        return 0.0
    device = next(model.parameters()).device
    total = 0
    correct = 0
    autocast_ctx = _autocast_context(device.type)
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        inputs, targets = _tensorize_batch(batch, tokenizer, MAX_SEQ_LEN, device)
        with autocast_ctx:
            logits = model(inputs)
        preds = logits.argmax(dim=-1)
        for row_idx in range(len(batch)):
            mask = targets[row_idx] != -1
            if mask.sum().item() == 0:
                continue
            total += 1
            if torch.equal(preds[row_idx][mask], targets[row_idx][mask]):
                correct += 1
    return correct / total if total else 0.0


def _generate_answer_ids(model, prompt_ids: list[int], tokenizer: Tokenizer,
                         last_move=None) -> list[int]:
    """Generate answer with single-token move prediction.

    Single forward pass to choose among 18 MOVE_face_turn tokens + DONE.
    Supports no-inverse rule: if last_move is given, masks the inverse token.
    """
    device = next(model.parameters()).device
    t2i = tokenizer.token_to_id
    autocast_ctx = _autocast_context(device.type)

    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    face_names = ("U", "R", "F", "D", "L", "B")
    turn_names = ("CW", "CCW", "HALF")
    done_id = t2i["<DONE>"]

    # Build valid token list (all moves + done, minus inverse)
    valid_ids = [done_id]
    for face in face_names:
        for turn in turn_names:
            if last_move and face == last_move.face:
                inv = {1: -1, -1: 1, 2: 2}[last_move.turns]
                inv_name = {1: "CW", -1: "CCW", 2: "HALF"}[inv]
                if turn == inv_name:
                    continue
            valid_ids.append(t2i[f"MOVE_{face}_{turn}"])

    with autocast_ctx:
        logits = model(input_ids)
    last_logits = logits[0, -1].float()
    mask = torch.full_like(last_logits, float('-inf'))
    for vid in valid_ids:
        mask[vid] = 0.0
    chosen = int((last_logits + mask).argmax().item())

    return [chosen]


def _build_prompt_ids(tokenizer: Tokenizer, cube: Cube, history: list[Move]) -> list[int]:
    prompt_tokens = build_prompt_tokens(cube.size, cube, history=history)
    return [tokenizer.get_bos_token_id(), *tokenizer.encode_tokens(prompt_tokens)]


def _enumerate_move_candidates(
    model,
    tokenizer: Tokenizer,
    cube: Cube,
    history: list[Move],
    visited_states: set[str],
) -> list[dict[str, object]]:
    device = next(model.parameters()).device
    t2i = tokenizer.token_to_id
    autocast_ctx = _autocast_context(device.type)

    face_names = ("U", "R", "F", "D", "L", "B")
    turn_names = ("CW", "CCW", "HALF")
    turn_to_val = {"CW": 1, "CCW": -1, "HALF": 2}
    last_move = history[-1] if history else None
    prompt_ids = _build_prompt_ids(tokenizer, cube, history)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    with autocast_ctx:
        logits = model(input_ids)
    last_logits = logits[0, -1].float()

    candidates: list[dict[str, object]] = []
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

            is_goal = next_cube.has_uniform_faces() if cube.size == 2 else next_cube.is_solved()
            candidates.append(
                {
                    "move": move,
                    "score": score,
                    "residual": _cube_residual_error(next_cube),
                    "next_cube": next_cube,
                    "next_state_str": next_state_str,
                    "is_goal": is_goal,
                }
            )
    return candidates


def _shortlist_candidates(candidates: list[dict[str, object]], current_residual: int) -> list[dict[str, object]]:
    acceptable = [
        candidate
        for candidate in candidates
        if candidate["residual"] <= current_residual + SEARCH_RESIDUAL_DELTA
    ]
    pool = acceptable if acceptable else candidates
    return sorted(pool, key=lambda candidate: candidate["score"], reverse=True)[:SEARCH_LOOKAHEAD_TOP_K]


@torch.no_grad()
def _select_move_greedy(
    model,
    tokenizer: Tokenizer,
    cube: Cube,
    history: list[Move],
    visited_states: set[str],
) -> Move | None:
    current_residual = _cube_residual_error(cube)
    candidates = _enumerate_move_candidates(model, tokenizer, cube, history, visited_states)
    if candidates:
        shortlist = _shortlist_candidates(candidates, current_residual)
        return shortlist[0]["move"]

    prompt_ids = _build_prompt_ids(tokenizer, cube, history)
    last_move = history[-1] if history else None
    answer_ids = _generate_answer_ids(model, prompt_ids, tokenizer, last_move=last_move)
    answer_tokens = tokenizer.decode_ids(answer_ids)
    try:
        return parse_answer_tokens(answer_tokens)
    except Exception:
        return None


@torch.no_grad()
def _select_move_two_ply(
    model,
    tokenizer: Tokenizer,
    cube: Cube,
    history: list[Move],
    visited_states: set[str],
) -> Move | None:
    """Select a move with deterministic 2-ply lookahead."""
    current_residual = _cube_residual_error(cube)
    candidates = _enumerate_move_candidates(model, tokenizer, cube, history, visited_states)

    if candidates:
        shortlist = _shortlist_candidates(candidates, current_residual)
        ranked: list[tuple[tuple[float, ...], Move]] = []
        for candidate in shortlist:
            if candidate["is_goal"]:
                ranked.append(
                    (
                        (0.0, 0.0, float(candidate["residual"]), -float(candidate["score"])),
                        candidate["move"],
                    )
                )
                continue

            next_history = [*history, candidate["move"]]
            next_visited = set(visited_states)
            next_visited.add(candidate["next_state_str"])
            second_candidates = _enumerate_move_candidates(
                model,
                tokenizer,
                candidate["next_cube"],
                next_history,
                next_visited,
            )

            if second_candidates:
                second_best = min(
                    second_candidates,
                    key=lambda item: (
                        0 if item["is_goal"] else 1,
                        item["residual"],
                        -item["score"],
                    ),
                )
                final_residual = second_best["residual"]
                solve_depth = 1.0 if second_best["is_goal"] else 2.0
                combined_score = candidate["score"] + SEARCH_SECOND_SCORE_DISCOUNT * second_best["score"]
            else:
                final_residual = candidate["residual"]
                solve_depth = 2.0
                combined_score = candidate["score"]

            ranked.append(
                (
                    (
                        solve_depth,
                        float(final_residual),
                        float(candidate["residual"]),
                        -float(combined_score),
                    ),
                    candidate["move"],
                )
            )

        ranked.sort(key=lambda item: item[0])
        return ranked[0][1]

    prompt_ids = _build_prompt_ids(tokenizer, cube, history)
    last_move = history[-1] if history else None
    answer_ids = _generate_answer_ids(model, prompt_ids, tokenizer, last_move=last_move)
    answer_tokens = tokenizer.decode_ids(answer_ids)
    try:
        return parse_answer_tokens(answer_tokens)
    except Exception:
        return None


@torch.no_grad()
def _select_move_value_guided(
    model,
    tokenizer: Tokenizer,
    cube: Cube,
    history: list[Move],
    visited_states: set[str],
) -> Move | None:
    """Select move using value head to evaluate candidate next-states.
    Combines residual filtering (for 2x2) with value-guided ranking."""
    candidates = _enumerate_move_candidates(model, tokenizer, cube, history, visited_states)
    if not candidates:
        return None

    # Immediate goal check
    for c in candidates:
        if c["is_goal"]:
            return c["move"]

    # Apply residual filter only for 2x2 (helps 2x2, hurts 3x3 value guidance)
    if cube.size == 2:
        current_residual = _cube_residual_error(cube)
        acceptable = [c for c in candidates if c["residual"] <= current_residual + SEARCH_RESIDUAL_DELTA]
        pool = acceptable if acceptable else candidates
    else:
        pool = candidates

    # If only one candidate after filtering, just use it
    if len(pool) == 1:
        return pool[0]["move"]

    device = next(model.parameters()).device
    autocast_ctx = _autocast_context(device.type)

    # Batch-evaluate candidate next-states with value head
    prompt_ids_list = []
    for c in pool:
        next_history = [*history, c["move"]]
        ids = _build_prompt_ids(tokenizer, c["next_cube"], next_history)
        prompt_ids_list.append(ids)

    max_len = max(len(ids) for ids in prompt_ids_list)
    pad_id = tokenizer.get_pad_token_id()
    batch = torch.full((len(prompt_ids_list), max_len), pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(prompt_ids_list):
        batch[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    with autocast_ctx:
        values = model.predict_value(batch).float()  # (N,) predicted distance-to-goal

    # Blend policy score and value prediction
    policy_scores = torch.tensor([c["score"] for c in pool], device=device)
    policy_probs = torch.softmax(policy_scores, dim=0)

    # Lower value = closer to goal = better
    alpha = 0.3  # policy weight (value-dominant)
    value_scores = -values
    if value_scores.std() > 1e-6:
        value_scores = (value_scores - value_scores.mean()) / value_scores.std()
    if policy_probs.std() > 1e-6:
        policy_norm = (policy_probs - policy_probs.mean()) / policy_probs.std()
    else:
        policy_norm = policy_probs
    combined = alpha * policy_norm + (1 - alpha) * value_scores

    best_idx = combined.argmax().item()
    return pool[best_idx]["move"]


@torch.no_grad()
def _select_move_with_search(
    model,
    tokenizer: Tokenizer,
    cube: Cube,
    history: list[Move],
    visited_states: set[str],
) -> Move | None:
    if SEARCH_SELECTOR == "hybrid_greedy_v1":
        return _select_move_greedy(model, tokenizer, cube, history, visited_states)
    if SEARCH_SELECTOR == "two_ply_v1":
        return _select_move_two_ply(model, tokenizer, cube, history, visited_states)
    if SEARCH_SELECTOR == "value_guided_v1":
        return _select_move_value_guided(model, tokenizer, cube, history, visited_states)
    if SEARCH_SELECTOR == "hybrid_auto_v1":
        # Best search per size: greedy+residual for 2x2, value-guided for 3x3+
        if cube.size == 2:
            return _select_move_greedy(model, tokenizer, cube, history, visited_states)
        return _select_move_value_guided(model, tokenizer, cube, history, visited_states)
    raise ValueError(f"Unsupported search selector: {SEARCH_SELECTOR}")


def _cube_residual_error(cube: Cube) -> int:
    """Count stickers that don't match their face's majority color.

    For 2x2 cubes where 'solved' means uniform faces (any global orientation),
    this is a better heuristic than checking against canonical colors.
    """
    from collections import Counter
    error = 0
    for face in ("U", "R", "F", "D", "L", "B"):
        colors = [c for row in cube.face_grid(face) for c in row]
        most_common_count = Counter(colors).most_common(1)[0][1]
        error += len(colors) - most_common_count
    return error


@torch.no_grad()
def _beam_search_solve(model, tokenizer: Tokenizer, cube: Cube, beam_width: int = 8, max_steps: int = 200) -> bool:
    """Beam search rollout: keep multiple partial solutions, expand the most promising."""
    device = next(model.parameters()).device
    autocast_ctx = _autocast_context(device.type)

    def is_goal(c):
        return c.has_uniform_faces() if c.size == 2 else c.is_solved()

    if is_goal(cube):
        return True

    # Each beam: (cube, history, visited, cumulative_value_score)
    initial_state = cube.to_kociemba_string()
    beams = [(cube.copy(), [], {initial_state}, 0.0)]

    for step in range(max_steps):
        if not beams:
            break

        all_candidates = []
        for beam_idx, (b_cube, b_history, b_visited, b_score) in enumerate(beams):
            if is_goal(b_cube):
                return True

            candidates = _enumerate_move_candidates(model, tokenizer, b_cube, b_history, b_visited)
            for c in candidates:
                if c["is_goal"]:
                    return True
                all_candidates.append((beam_idx, c))

        if not all_candidates:
            break

        # Batch evaluate all candidate next-states with value head
        prompt_ids_list = []
        for beam_idx, c in all_candidates:
            b_history = beams[beam_idx][1]
            next_history = [*b_history, c["move"]]
            ids = _build_prompt_ids(tokenizer, c["next_cube"], next_history)
            prompt_ids_list.append(ids)

        max_len = max(len(ids) for ids in prompt_ids_list)
        pad_id = tokenizer.get_pad_token_id()
        batch = torch.full((len(prompt_ids_list), max_len), pad_id, dtype=torch.long, device=device)
        for i, ids in enumerate(prompt_ids_list):
            batch[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

        with autocast_ctx:
            values = model.predict_value(batch).float()

        # Score: blend policy logit + negative value (lower distance = better)
        scored = []
        for i, (beam_idx, c) in enumerate(all_candidates):
            parent_score = beams[beam_idx][3]
            # Value = predicted distance to goal (lower is better)
            candidate_score = parent_score + c["score"] - 0.5 * values[i].item()
            scored.append((candidate_score, beam_idx, c, values[i].item()))

        # Keep top beam_width candidates
        scored.sort(key=lambda x: -x[0])  # highest score first
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
            new_beams.append((c["next_cube"], new_history, new_visited, score))

        beams = new_beams

    # Check final beams
    for b_cube, _, _, _ in beams:
        if is_goal(b_cube):
            return True
    return False


@torch.no_grad()
def evaluate_rollouts(model, tokenizer: Tokenizer, episodes: list[Episode]) -> tuple[float, dict[int, dict[str, float]]]:
    if not episodes:
        return 0.0, {}

    solved = 0
    per_size = defaultdict(lambda: {"count": 0, "solved": 0, "invalid": 0, "residual_sum": 0.0})

    def is_goal(cube: Cube, size: int) -> bool:
        if size == 2:
            return cube.has_uniform_faces()
        return cube.is_solved()

    for episode in episodes:
        cube = Cube(episode.size)
        cube.apply_moves(episode.scramble)
        invalid = False
        moves_taken: list[Move] = []
        visited_states = {cube.to_kociemba_string()}
        for _ in range(max(episode.max_rollout_steps, ROLLOUT_MIN_STEPS)):
            if is_goal(cube, episode.size):
                break
            move = _select_move_with_search(
                model,
                tokenizer,
                cube,
                moves_taken,
                visited_states,
            )
            if move is None:
                break
            try:
                cube.apply_move(move)
            except Exception:
                invalid = True
                break
            moves_taken.append(move)
            visited_states.add(cube.to_kociemba_string())

        is_solved = is_goal(cube, episode.size)
        solved += int(is_solved)
        bucket = per_size[episode.size]
        bucket["count"] += 1
        bucket["solved"] += int(is_solved)
        bucket["invalid"] += int(invalid)
        bucket["residual_sum"] += 0.0 if is_solved else _cube_residual_error(cube)

    size_metrics: dict[int, dict[str, float]] = {}
    for size, stats in per_size.items():
        size_metrics[size] = {
            "count": stats["count"],
            "solve_rate": stats["solved"] / stats["count"],
            "invalid_rate": stats["invalid"] / stats["count"],
            "mean_residual": stats["residual_sum"] / stats["count"],
        }
    return solved / len(episodes), size_metrics


@torch.no_grad()
def evaluate_policy(model, tokenizer: Tokenizer, batch_size: int) -> dict[str, object]:
    payload = load_dataset()
    episodes = payload["eval_episodes"]
    id_episodes = [Episode.from_dict(item) for item in episodes["id"]]
    ood_dev_episodes = [Episode.from_dict(item) for item in episodes["ood_dev"]]
    ood_test_episodes = [Episode.from_dict(item) for item in episodes["ood_test"]]

    id_move_accuracy = evaluate_move_accuracy(model, tokenizer, payload["id_val_examples"], batch_size)
    ood_dev_move_accuracy = evaluate_move_accuracy(model, tokenizer, payload["ood_dev_examples"], batch_size)

    id_solve_rate, id_size_metrics = evaluate_rollouts(model, tokenizer, id_episodes)
    ood_dev_solve_rate, ood_dev_size_metrics = evaluate_rollouts(model, tokenizer, ood_dev_episodes)
    ood_test_solve_rate, ood_test_size_metrics = evaluate_rollouts(model, tokenizer, ood_test_episodes)
    primary_metric = ood_dev_solve_rate if ood_dev_episodes else id_solve_rate

    return {
        "primary_metric": primary_metric,
        "id_move_accuracy": id_move_accuracy,
        "ood_dev_move_accuracy": ood_dev_move_accuracy,
        "id_solve_rate": id_solve_rate,
        "ood_dev_solve_rate": ood_dev_solve_rate,
        "ood_test_solve_rate": ood_test_solve_rate,
        "size_metrics": {
            "id": id_size_metrics,
            "ood_dev": ood_dev_size_metrics,
            "ood_test": ood_test_size_metrics,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Prepare Rubik's-cube policy data")
    parser.add_argument("--force", action="store_true", help="Regenerate tokenizer and dataset")
    args = parser.parse_args()

    report_environment()
    print(f"Cache directory: {CACHE_DIR}")
    print()

    ensure_tokenizer(force=args.force)
    build_dataset_payload(force=args.force)
    print()
    print("Done! Ready to train.")


if __name__ == "__main__":
    main()
