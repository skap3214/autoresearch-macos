"""
Expert Iteration (ExIt) fine-tuning for 3x3 Rubik's cube.

Warm-starts from exp 3 checkpoint, fine-tunes on 50/50 mix of
teacher data and beam-search-distilled ExIt data.

Usage: python train_exit.py [--exit-data exit_examples.pkl] [--checkpoint exp3_model.pt]
"""

import os, sys, time, pickle, argparse, random, contextlib
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rubiks-cube-NxNxN-solver'))

from prepare import (
    build_vocab, Tokenizer, load_dataset, evaluate_rollouts,
    encode_supervised_example, MAX_SEQ_LEN,
    _augment_example_online, _get_augmentation_tables,
)
from rubiks import Episode, Cube, build_prompt_tokens, build_answer_tokens
from playground import load_checkpoint

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE_BATCH_SIZE = 512
TRAIN_SEQ_LEN = 66       # exp 3 used 66
TIME_BUDGET = 3600        # 1 hour fine-tuning (stop early if degrading)
EXIT_MIX_RATIO = 0.01     # fraction of ExIt data in each batch (5%)
LR = 3e-5                 # low LR for fine-tuning (1/10 of original)
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.05
EVAL_EVERY_SECS = 300     # eval every 5 min
EVAL_CUBES = 64
CHECKPOINT_EVERY_SECS = 600
VALUE_HEAD_WEIGHT = 0.5
VALUE_BIAS = 5.0

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def build_exp3_tokenizer():
    """Build the 82-token tokenizer matching exp 3's vocab."""
    all_vocab = build_vocab()
    vocab = [t for t in all_vocab if not t.startswith("WMOVE_") and not t.startswith("STAGE_444_")]
    return Tokenizer(
        token_to_id={t: i for i, t in enumerate(vocab)},
        id_to_token=vocab,
    )


def load_teacher_examples():
    """Load pre-encoded teacher examples, filtering to 2x2+3x3 and remapping token IDs.

    The dataset was encoded with the 105-token vocab where STAGE tokens are at 95-99.
    The exp 3 model uses an 82-token vocab where STAGE tokens are at 77-81.
    We remap STAGE token IDs and filter out 4x4+ examples.
    """
    payload = load_dataset()
    examples = payload["train_examples"]

    # Build remap: 105-tok STAGE IDs -> 82-tok STAGE IDs
    # 95->77, 96->78, 97->79, 98->80, 99->81
    remap = {95: 77, 96: 78, 97: 79, 98: 80, 99: 81}

    result = []
    for ex in examples:
        if ex.get("size", 3) > 3:
            continue
        # Check if any token needs remapping (3x3 examples with STAGE tokens)
        needs_remap = any(tid in remap for tid in ex["input_ids"])
        if needs_remap:
            ex = dict(ex)  # shallow copy
            ex["input_ids"] = [remap.get(tid, tid) for tid in ex["input_ids"]]
            ex["targets"] = [remap.get(tid, tid) if tid >= 0 else tid for tid in ex["targets"]]
        result.append(ex)
    return result


def example_stream(examples, rng):
    """Infinite shuffled stream."""
    epoch = 0
    while True:
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        for idx in indices:
            yield examples[idx], epoch
        epoch += 1


def make_batched_loader(teacher_stream, exit_stream, batch_size, seq_len, device, tokenizer):
    """Interleave teacher and ExIt examples in each batch with online augmentation."""
    T = seq_len
    B = batch_size
    pad_id = tokenizer.token_to_id.get("<|pad|>", 0)

    n_exit = int(B * EXIT_MIX_RATIO)
    n_teacher = B - n_exit

    # Online symmetry augmentation
    sticker_perms, move_maps, color_maps = _get_augmentation_tables(tokenizer)
    aug_rng = random.Random(12345)

    cpu_inputs = torch.zeros(B, T, dtype=torch.long)
    cpu_targets = torch.full((B, T), -1, dtype=torch.long)
    cpu_distances = torch.zeros(B, dtype=torch.float32)

    inputs = torch.zeros(B, T, dtype=torch.long, device=device)
    targets = torch.full((B, T), -1, dtype=torch.long, device=device)
    distances = torch.zeros(B, dtype=torch.float32, device=device)

    while True:
        for row_idx in range(B):
            if row_idx < n_teacher:
                example, epoch = next(teacher_stream)
            else:
                example, epoch = next(exit_stream)

            input_ids = example["input_ids"]
            target_ids = example["targets"]

            # Online symmetry augmentation (random rotation)
            size = example.get("size", 3)
            rot_idx = aug_rng.randint(0, 23)
            if size in sticker_perms:
                input_ids, target_ids = _augment_example_online(
                    input_ids, target_ids, size, rot_idx,
                    sticker_perms, move_maps, color_maps
                )

            seq_len_ex = min(len(input_ids), T)
            cpu_inputs[row_idx].fill_(pad_id)
            cpu_targets[row_idx].fill_(-1)
            cpu_inputs[row_idx, :seq_len_ex] = torch.tensor(input_ids[:seq_len_ex], dtype=torch.long)
            cpu_targets[row_idx, :seq_len_ex] = torch.tensor(target_ids[:seq_len_ex], dtype=torch.long)
            cpu_distances[row_idx] = float(example.get("distance_to_goal", 0))

        inputs.copy_(cpu_inputs, non_blocking=True)
        targets.copy_(cpu_targets, non_blocking=True)
        distances.copy_(cpu_distances, non_blocking=True)
        yield inputs, targets, distances, epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="exp3_model.pt")
    parser.add_argument("--exit-data", default="exit_examples.pkl")
    parser.add_argument("--time-budget", type=int, default=TIME_BUDGET)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build exp 3-compatible tokenizer
    tokenizer = build_exp3_tokenizer()
    print(f"Tokenizer: {tokenizer.get_vocab_size()} tokens")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, device=device)
    model.train()
    print(f"Model: {model.config.n_layer}L, {model.config.n_embd}D, "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    # Load ExIt data
    print(f"Loading ExIt data: {args.exit_data}")
    with open(args.exit_data, "rb") as f:
        exit_examples = pickle.load(f)
    print(f"ExIt examples: {len(exit_examples)}")

    # Load teacher data
    print("Loading teacher data...")
    teacher_examples = load_teacher_examples()
    print(f"Teacher examples: {len(teacher_examples)}")

    # Create streams
    rng = random.Random(42)
    teacher_stream = example_stream(teacher_examples, random.Random(42))
    exit_stream = example_stream(exit_examples, random.Random(43))

    # Create batched loader
    loader = make_batched_loader(
        teacher_stream, exit_stream,
        DEVICE_BATCH_SIZE, TRAIN_SEQ_LEN, device, tokenizer,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    # Autocast
    autocast_ctx = torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()

    # Run dir
    from datetime import datetime
    from pathlib import Path
    run_dir = Path("runs") / f"exit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # wandb
    try:
        import wandb
        wandb.init(project="rubiks-cube-solver", name=f"exit_{run_dir.name}",
                   config={"checkpoint": args.checkpoint, "exit_data": args.exit_data,
                           "exit_examples": len(exit_examples), "teacher_examples": len(teacher_examples),
                           "lr": args.lr, "batch_size": DEVICE_BATCH_SIZE,
                           "mix_ratio": EXIT_MIX_RATIO, "time_budget": args.time_budget})
        use_wandb = True
    except Exception:
        use_wandb = False

    # Load eval episodes
    payload = load_dataset()
    eval_eps = [Episode.from_dict(e) for e in payload["eval_episodes"]["id"]]
    eval_3x3 = [e for e in eval_eps if e.size == 3]
    eval_2x2 = [e for e in eval_eps if e.size == 2]
    print(f"Eval: {len(eval_2x2)} 2x2, {len(eval_3x3)} 3x3")

    # Prefetch first batch
    x, y, d, epoch = next(loader)

    # Training loop
    t_start = time.time()
    step = 0
    smooth_loss = 0.0
    best_sr_3x3 = 0.0
    last_eval_time = 0
    last_ckpt_time = 0

    print(f"\nFine-tuning for {args.time_budget}s with {EXIT_MIX_RATIO:.0%} ExIt data...")
    print(f"LR: {args.lr}, Batch: {DEVICE_BATCH_SIZE}")
    print()

    while True:
        t0 = time.time()

        with autocast_ctx:
            loss = model(x, y, distances=d)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        x, y, d, epoch = next(loader)

        dt = time.time() - t0
        elapsed = time.time() - t_start

        ema = 0.95
        smooth_loss = ema * smooth_loss + (1 - ema) * loss.item()
        debiased = smooth_loss / (1 - ema ** (step + 1))

        if step % 50 == 0:
            remaining = args.time_budget - elapsed
            print(f"\rstep {step:06d} | loss: {debiased:.4f} | dt: {dt*1000:.0f}ms | "
                  f"epoch: {epoch} | remaining: {remaining:.0f}s    ",
                  end="", flush=True)
            if use_wandb and step % 200 == 0:
                wandb.log({"train/loss": debiased}, step=step)

        # Eval
        if elapsed - last_eval_time >= EVAL_EVERY_SECS:
            last_eval_time = elapsed
            model.eval()

            # Quick greedy eval
            sr_2x2, _ = evaluate_rollouts(model, tokenizer, eval_2x2[:32])
            sr_3x3, _ = evaluate_rollouts(model, tokenizer, eval_3x3[:EVAL_CUBES])

            print(f"\n  Eval @ step {step}: 2x2={sr_2x2:.0%} | 3x3={sr_3x3:.0%}")

            if use_wandb:
                wandb.log({"eval/2x2_greedy": sr_2x2, "eval/3x3_greedy": sr_3x3}, step=step)

            if sr_3x3 > best_sr_3x3:
                best_sr_3x3 = sr_3x3
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": model.config,
                    "step": step,
                    "loss": debiased,
                    "sr_3x3": sr_3x3,
                }, run_dir / "model_best.pt")
                print(f"  [BEST 3x3={sr_3x3:.0%}, saved]")

            model.train()

        # Checkpoint
        if elapsed - last_ckpt_time >= CHECKPOINT_EVERY_SECS:
            last_ckpt_time = elapsed
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model.config,
                "step": step,
                "loss": debiased,
            }, run_dir / "model_latest.pt")

        step += 1
        if elapsed >= args.time_budget:
            break

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.config,
        "step": step,
        "loss": debiased,
    }, run_dir / "model_final.pt")

    print(f"\n\nExIt fine-tuning complete: {step} steps in {elapsed:.0f}s")
    print(f"Best 3x3 greedy: {best_sr_3x3:.0%}")

    # Final eval
    model.eval()
    sr_2x2, _ = evaluate_rollouts(model, tokenizer, eval_2x2)
    sr_3x3, _ = evaluate_rollouts(model, tokenizer, eval_3x3)
    print(f"Final eval: 2x2={sr_2x2:.0%} | 3x3={sr_3x3:.0%}")

    if use_wandb:
        wandb.log({"final/2x2_greedy": sr_2x2, "final/3x3_greedy": sr_3x3})
        wandb.finish()


if __name__ == "__main__":
    main()
