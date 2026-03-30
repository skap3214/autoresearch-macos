"""
Expert Iteration (ExIt) data generation for 3x3 Rubik's cube.

Uses beam search to solve scrambled cubes, then converts the solved
trajectories into supervised training examples. These are mixed with
original teacher data for fine-tuning.

Usage: python generate_exit_data.py [--num-scrambles 10000] [--beam-width 32]
"""

import os, sys, pickle, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rubiks-cube-NxNxN-solver'))

import torch
from playground import load_checkpoint
from prepare import (
    Tokenizer, build_vocab, load_dataset, _beam_search_solve_with_trace,
    encode_supervised_example, MAX_SEQ_LEN,
)
from rubiks import (
    Cube, Move, Episode, random_scramble, scramble_length_for_size,
    build_prompt_tokens, build_answer_tokens,
)
import random


def generate_exit_examples(model, tokenizer, num_scrambles=10000,
                           beam_width=32, max_steps=200, seed=42):
    """Generate ExIt training examples by solving cubes with beam search.

    Returns list of supervised examples in the same format as teacher data.
    """
    rng = random.Random(seed)
    examples = []
    solved_count = 0
    total_moves = 0

    print(f"Generating ExIt data: {num_scrambles} scrambles, beam_width={beam_width}")

    for i in range(num_scrambles):
        # Generate random 3x3 scramble
        scramble = random_scramble(3, scramble_length_for_size(3), rng=rng)
        cube = Cube(3)
        cube.apply_moves(scramble)

        # Try to solve with beam search (returns move trace)
        solved, moves = _beam_search_solve_with_trace(
            model, tokenizer, cube, beam_width=beam_width, max_steps=max_steps
        )

        if solved and len(moves) > 0:
            solved_count += 1
            total_moves += len(moves)

            # Convert to training examples (same format as teacher data)
            replay_cube = Cube(3)
            replay_cube.apply_moves(scramble)
            history = []

            for j, move in enumerate(moves):
                prompt_tokens = build_prompt_tokens(3, replay_cube, history=history)
                answer_tokens = build_answer_tokens(move)
                distance = len(moves) - j

                try:
                    encoded = encode_supervised_example(tokenizer, prompt_tokens, answer_tokens)
                    encoded["size"] = 3
                    encoded["distance_to_goal"] = distance
                    examples.append(encoded)
                except (ValueError, AssertionError):
                    continue

                replay_cube.apply_move(move)
                history.append(move)

            # Add DONE token
            prompt_tokens = build_prompt_tokens(3, replay_cube, history=history)
            answer_tokens = build_answer_tokens(None)
            try:
                encoded = encode_supervised_example(tokenizer, prompt_tokens, answer_tokens)
                encoded["size"] = 3
                encoded["distance_to_goal"] = 0
                examples.append(encoded)
            except (ValueError, AssertionError):
                pass

        if (i + 1) % 100 == 0:
            sr = solved_count / (i + 1)
            print(f"  {i+1}/{num_scrambles}: solved {solved_count} ({sr:.0%}), "
                  f"{len(examples)} examples, avg_moves={total_moves/max(1,solved_count):.1f}")

    print(f"\nExIt generation complete:")
    print(f"  Solved: {solved_count}/{num_scrambles} ({solved_count/num_scrambles:.0%})")
    print(f"  Examples: {len(examples)}")
    print(f"  Avg solution length: {total_moves/max(1,solved_count):.1f} moves")

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="exp3_model.pt")
    parser.add_argument("--num-scrambles", type=int, default=10000)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output", default="exit_examples.pkl")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, device=device)
    print(f"Model: {model.config.n_layer}L, {model.config.n_embd}D, vocab={model.config.vocab_size}")

    # Build tokenizer matching model vocab
    # The exp 3 model (82 tokens) was trained before WMOVE tokens were added.
    # Its vocab is the current vocab minus WMOVE and 4x4 STAGE tokens.
    all_vocab = build_vocab()
    if model.config.vocab_size < len(all_vocab):
        # Filter out tokens that weren't in the original vocab
        vocab = [t for t in all_vocab if not t.startswith("WMOVE_") and not t.startswith("STAGE_444_")]
        vocab = vocab[:model.config.vocab_size]
    else:
        vocab = all_vocab
    tokenizer = Tokenizer(
        token_to_id={t: i for i, t in enumerate(vocab)},
        id_to_token=vocab,
    )
    print(f"Tokenizer: {tokenizer.get_vocab_size()} tokens")

    # Generate ExIt examples
    t0 = time.time()
    examples = generate_exit_examples(
        model, tokenizer,
        num_scrambles=args.num_scrambles,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
    )
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "wb") as f:
        pickle.dump(examples, f)
    print(f"Saved {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
