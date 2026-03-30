# Rubik's Cube Neural Solver — Full Handoff Summary

## Goal
Train a **single neural network** that can completely solve Rubik's cubes from **2x2 through 7x7**. Rigorous ML methodology required — no shortcuts.

## Hardware
- Lambda A100-SXM4-40GB, 30 CPU cores, 221GB RAM
- System Python 3.12, torch 2.7.0
- dwalton NxN solver at /home/ubuntu/rubiks-cube-NxNxN-solver

## Repository
- `/home/ubuntu/autoresearch-macos` (branch: rubiks-2x2-solver)
- HuggingFace: soamikapadia/rubiks-2x2-solver

## Architecture
- GPT-style transformer: D=12, 768 dim, 12 heads, 85.4M params
- Input: flat sticker colors + CFOP stage token (3x3) + last 3 move history
- Output: single token from 37-class vocab (18 MOVE + 18 WMOVE + DONE)
- Auxiliary value head predicting distance-to-goal
- MuonAdamW optimizer, online 24x symmetry augmentation
- Vocab: 100 tokens total (was 82 before WMOVE addition)

## Key Files
- `train.py` — model, optimizer, training loop, wandb logging
- `prepare.py` — data gen, tokenizer, evaluation, beam search, online augmentation
- `rubiks.py` — NxN cube simulator, CFOP detection, symmetry transforms, move/episode transforms
- `teacher_dwalton.py` — dwalton solver wrapper (2x2, 3x3, 4x4)
- `test_beam.py` — standalone beam search eval
- `results.tsv` — experiment log (60+ experiments)
- `docs/plans/v3_a100-experiments-and-methodology.md` — methodology doc

## Experiment Results

### Exp 1 (2x2+3x3, batch 256, 6hr, 262K 3x3 episodes)
- Greedy: 2x2 82%, 3x3 18%
- Beam w=8: 2x2 97%, 3x3 72%
- Early stopped at 50% due to overfitting

### Exp 2 (2x2+3x3, batch 1024, 24hr, no early stop)
- Greedy: 2x2 99.2% (256 cubes), 3x3 20.7% (256 cubes)
- Beam w=8: 2x2 100%, 3x3 64.8% (256 cubes)
- Beam w=32: 3x3 95.3% (64 cubes)
- Beam w=64: 3x3 100% (64 cubes)
- Severe overfitting by epoch 8, killed at step 75K

### Exp 3 (2x2+3x3, batch 1024, 12hr, online 24x symmetry augmentation) ← BEST 2x2+3x3
- Greedy: 2x2 100%, 3x3 34% (256 cubes)
- Beam w=8: 2x2 100%, 3x3 64% (256 cubes)
- Beam w=64: 2x2 100%, 3x3 97% (64 cubes)
- Val loss 0.809, ZERO overfitting through 50 epochs
- Checkpoint: runs/20260326_015858/model.pt and model_best.pt
- HuggingFace: model-joint-2x2-3x3-v3-symm-aug.pt

### Exp 4 (2x2+3x3+4x4 joint, batch 512, 12hr)
- REGRESSION: adding 4x4 degraded everything
- Greedy: 2x2 86%, 3x3 11%, 4x4 0%
- Beam w=64: 3x3 67%, 4x4 0%
- 85M model insufficient capacity for 3 sizes
- Only 16K 4x4 episodes (solver too slow)

### Exp 5 (4x4-only, self-supervised, batch 512, 6hr)
- Self-supervised: scramble-inverse (no solver needed)
- 262K episodes, 6M training examples
- Val loss: 3.6 → 2.18 (significant learning)
- Move accuracy: 59%
- Greedy: 0%, Beam w=8: 0%
- Model learned 4x4 patterns but not enough to solve

## Key Innovations Implemented
1. **Online symmetry augmentation**: 24x effective data, zero overhead. Precomputed sticker permutation + color relabeling tables. Applied randomly per example in dataloader. Verified correct with 144/144 tests.
2. **WMOVE tokens**: 18 wide move tokens for 4x4+ (width=2 moves like Uw, Rw)
3. **Self-supervised data gen**: scramble from solved, use inverse as solution. Instant, unlimited data.
4. **wandb integration**: real-time experiment tracking

## What Works
- DAgger mid-training (biggest single improvement for 2x2 historically)
- Auxiliary value head (weight=0.5) — 3.5x multiplier
- Joint MOVE tokens (single-token prediction)
- Kociemba teacher for 3x3 (short solutions)
- Online symmetry augmentation (eliminates overfitting)
- Beam search (compensates for greedy weakness)

## What Doesn't Work
- Curriculum (mixed scramble lengths) — hurts
- Weight decay > 0 — hurts
- Value loss weight > 0.5 — degrades policy
- D=16 (202M) at short time budgets — underfits
- Joint 2x2+3x3+4x4 training with 85M model — insufficient capacity
- Self-supervised 4x4 training alone — learns patterns but can't solve

## Known Issues
1. CFOP stage tokens applied to 4x4 (wrong — CFOP is 3x3-specific)
2. Value targets are total distance-to-goal (too noisy for 4x4's 40-move horizon)
3. No 4x4 progress metrics (centers done, edges paired, etc.)
4. The dwalton 4x4 solver is very slow (~5-10s per cube)

## Current Plan (agreed upon)
The next milestone is a **4x4 compatibility milestone**, NOT another training run:

1. **Disable CFOP stage tokens for size > 3** — wrong abstraction, actively harmful
2. **Add 4x4 as OOD_DEV** — evaluate zero-shot transfer from 2x2+3x3 model
3. **Implement 4x4 progress metrics** — centers_done_rate, edges_paired_rate, reduced_to_3x3_rate, mean_stage_reached
4. **Zero-shot eval** of exp 3 model on 4x4 with beam search
5. **Then**: implement reduction-stage detection and stage-local value targets
6. **Then**: retrain with proper 4x4 integration

## Methodology Principles
- No shortcuts — rigorous ML best practices
- One variable at a time — proper ablation
- 256+ cube evals with confidence intervals
- Greedy solve rate is the primary metric (beam search masks weakness)
- Never start training without a clear hypothesis for improvement
- Stop training when progress plateaus — don't waste GPU time
- Always parallelize independent work

## Research Context
- **EfficientCube**: Policy-only + beam search, SOTA on 3x3 with self-supervised training
- **CayleyPy** (NeurIPS 2025): First ML to solve 4x4/5x5, uses diffusion distance + beam search
- **No published work** trains a single model across multiple cube sizes — this is novel
- Symmetry augmentation is unexploited in the literature (we implemented it)
