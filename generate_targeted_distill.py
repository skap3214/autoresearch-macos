"""
Targeted first-move distillation for 3x3 Rubik's cube.

For each scramble:
1. Run greedy to get greedy's first move
2. Run beam search (w=64) to get beam's first move + solution
3. Keep only disagreement states where beam solves
4. Create single supervised example: state → beam's first move

Usage: python generate_targeted_distill.py [--num-scrambles 2000]
"""

import os, sys, pickle, time, random
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rubiks-cube-NxNxN-solver'))

import torch
from playground import load_checkpoint
from prepare import (
    Tokenizer, build_vocab, _beam_search_solve_with_trace,
    _enumerate_move_candidates, encode_supervised_example,
)
from rubiks import (
    Cube, random_scramble, scramble_length_for_size,
    build_prompt_tokens, build_answer_tokens,
)


def build_exp3_tokenizer():
    all_vocab = build_vocab()
    vocab = [t for t in all_vocab if not t.startswith("WMOVE_") and not t.startswith("STAGE_444_")]
    return Tokenizer(token_to_id={t: i for i, t in enumerate(vocab)}, id_to_token=vocab)


def get_greedy_first_move(model, tokenizer, cube):
    """Get the greedy policy's top move for a cube state."""
    candidates = _enumerate_move_candidates(model, tokenizer, cube, [], set())
    if not candidates:
        return None
    # Sort by policy score (highest first)
    candidates.sort(key=lambda c: -c["score"])
    return candidates[0]["move"]


def generate_targeted_examples(model, tokenizer, num_scrambles=2000,
                                beam_width=8, max_steps=100, seed=42):
    """Generate targeted distillation examples at beam/greedy disagreement states."""
    rng = random.Random(seed)
    examples = []
    stats = {
        "total": 0, "beam_solved": 0, "greedy_agrees": 0,
        "disagreement": 0, "too_long": 0, "kept": 0,
    }

    print(f"Targeted distillation: {num_scrambles} scrambles, beam_w={beam_width}")

    for i in range(num_scrambles):
        stats["total"] += 1
        scramble = random_scramble(3, scramble_length_for_size(3), rng=rng)
        cube = Cube(3)
        cube.apply_moves(scramble)

        # Get greedy's first move
        greedy_move = get_greedy_first_move(model, tokenizer, cube)

        # Get beam's solution
        solved, beam_moves = _beam_search_solve_with_trace(
            model, tokenizer, cube, beam_width=beam_width, max_steps=max_steps
        )

        if not solved or len(beam_moves) == 0:
            continue
        stats["beam_solved"] += 1

        # Filter: solution not too long (within teacher range)
        if len(beam_moves) > 22:
            stats["too_long"] += 1
            continue

        beam_first_move = beam_moves[0]

        # Check disagreement
        if greedy_move is not None and beam_first_move.face == greedy_move.face and beam_first_move.turns == greedy_move.turns:
            stats["greedy_agrees"] += 1
            continue

        stats["disagreement"] += 1

        # Create supervised example: state → beam's first move
        prompt_tokens = build_prompt_tokens(3, cube, history=[])
        answer_tokens = build_answer_tokens(beam_first_move)

        try:
            encoded = encode_supervised_example(tokenizer, prompt_tokens, answer_tokens)
            encoded["size"] = 3
            encoded["distance_to_goal"] = len(beam_moves)
            examples.append(encoded)
            stats["kept"] += 1
        except (ValueError, AssertionError):
            continue

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{num_scrambles}: beam_solved={stats['beam_solved']} "
                  f"disagree={stats['disagreement']} kept={stats['kept']} "
                  f"agree={stats['greedy_agrees']} too_long={stats['too_long']}")

    print(f"\nTargeted distillation complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return examples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="exp3_model.pt")
    parser.add_argument("--num-scrambles", type=int, default=2000)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output", default="targeted_distill.pkl")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint, device="cuda")
    tokenizer = build_exp3_tokenizer()
    print(f"Tokenizer: {tokenizer.get_vocab_size()} tokens")

    t0 = time.time()
    examples = generate_targeted_examples(
        model, tokenizer,
        num_scrambles=args.num_scrambles,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
    )
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "wb") as f:
        pickle.dump(examples, f)
    print(f"Saved {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
