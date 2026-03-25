# A100 Experiment Log & Methodology Review (v3)

## Updated Goal

Train a **single neural network** that can completely solve Rubik's cubes from **2x2 through 7x7**. This is an AI research project — rigorous methodology required.

## Current Best Results

Best model: `model-joint-2x2-3x3-v2.pt` (HuggingFace: `soamikapadia/rubiks-2x2-solver`)
- D=12, 768 dim, 85.4M params
- Trained on 65K 2x2 + 262K 3x3 Kociemba episodes
- Checkpoint from exp 2, step 36K (best val loss = 0.750)

| Metric | Result | Eval size |
|--------|--------|-----------|
| 2x2 greedy | 100% | 32 cubes |
| 3x3 greedy | ~28% | 32 cubes |
| 2x2 beam w=8 | 100% | 64 cubes |
| 3x3 beam w=8 | 67% | 64 cubes |
| 2x2 beam w=32 | 100% | 64 cubes |
| 3x3 beam w=32 | 95% | 64 cubes |

**Note:** These eval sizes are too small for reliable conclusions. Future evals must use 256+ cubes.

## Experiment History

### Experiment 1: More data + more time
- **Config:** D=12, batch 256, TIME_BUDGET=21600 (6hr), 262K 3x3 episodes, PATIENCE=5
- **Result:** Early stopped at step 130K (50%). Greedy: 2x2:82% 3x3:18%. Beam w=8: 2x2:97% 3x3:72%.
- **Key finding:** 3x3 greedy improved from 11% (d12v8) to 18%. But beam w=8 regressed from 84% to 72%.
- **Hypothesis:** Early stopping prevented full convergence; model needed more epochs.

### Experiment 2: Batch 1024, no early stopping
- **Config:** D=12, batch 1024, TIME_BUDGET=86400 (24hr), 262K 3x3, PATIENCE=999
- **Result:** Stopped at step 75K due to severe overfitting (val-train gap >1.0). Greedy: 2x2:100% 3x3:28%. Beam w=32: 3x3:95%.
- **Key finding:** Larger batch converges faster but overfits earlier. Val loss decoupled from solve rate — model kept improving solve rates despite worsening val loss. Eventually solve rates degraded too (3x3 dropped to 6% at step 70K).
- **GPU utilization:** 25GB/40GB VRAM, 44% MFU (up from 33.9% at batch 256).

## Methodology Issues Identified

These must be fixed in all future experiments:

### 1. Evaluation sample size too small
Quick evals use 32 cubes, beam evals use 64 cubes. At 64 cubes, a 95% result has a 95% CI of roughly 87-99%. The variance is too high for reliable comparisons.
- **Fix:** 256+ cubes for all reported results. Report confidence intervals.

### 2. No LR sweep when changing batch size
We 4x'd batch size (256→1024) but kept LR=0.12 unchanged. Standard practice requires LR scaling (linear, sqrt, or sweep). The rapid overfitting may be partly caused by this mismatch.
- **Fix:** LR sweep when changing batch size. Document the scaling rule used.

### 3. No proper ablation
Exp 1→2 changed batch size, time budget, AND early stopping simultaneously. We can't isolate which change helped.
- **Fix:** Change one variable at a time. Maintain an ablation table.

### 4. Beam search masks model weakness
Greedy 3x3 is only 28% — the model gets the wrong move 72% of the time. Beam w=32 compensates by exploring 32 paths per step. This won't scale to 7x7 where solutions are 200+ moves.
- **Fix:** Track greedy solve rate as primary metric. Beam search is secondary.

### 5. Val loss diverged from solve rate
We observed val loss worsening while solve rates improved, then eventually solve rates degraded too. We noted the anomaly but didn't investigate.
- **Fix:** Implement solve-rate-based checkpointing. Investigate why CE loss decouples from task performance (likely because loss weights all positions equally, but only critical branching points matter for solve rate).

### 6. Imitation learning has scaling concerns
Pure behavioral cloning from a solver suffers from compounding errors. For 7x7 (294 stickers, 200+ moves), the error compounds over a much longer horizon.
- **Fix:** Consider RL fine-tuning (PPO/REINFORCE with solve success as reward) after imitation pre-training. Or planning-based approaches (MCTS + policy network, AlphaZero-style).

## What Worked (keep these)

- **DAgger mid-training** — biggest single improvement for 2x2 historically
- **Auxiliary value head (weight=0.5)** — 3.5x multiplier, confirmed by ablation
- **Joint MOVE tokens** — single-token prediction beats sequential FACE→TURN
- **Kociemba teacher for 3x3** — shorter solutions than CFOP, easier to learn
- **More data + more time together** — neither alone works
- **Batch 1024 on A100** — 44% MFU vs 34% at batch 256, better GPU utilization

## What Doesn't Work (don't retry)

- Curriculum (mixed scramble lengths) — hurts
- Weight decay > 0 — hurts
- Value loss weight > 0.5 — degrades policy
- SEARCH_RESIDUAL_DELTA = 1 (0%) or 3 (worse than 2)
- D=16 (202M params) — underfits without enough steps
- Value-guided search replacing residual for greedy — value not accurate enough

## Next Steps

### Phase 1: Solidify 3x3
1. Test wider beam (w=64, w=128) on current model — free performance check
2. Proper 256-cube evaluation of current model with confidence intervals
3. DAgger experiment with proper LR tuning
4. Investigate solve-rate-based checkpointing

### Phase 2: Add 4x4
1. Extend move parser for wide moves (Uw, Lw, etc.)
2. Add `solve_cube_444/555/666/777` wrappers
3. Increase MAX_SEQ_LEN (104 for 4x4, 302 for 7x7)
4. Parameterize ValueMLP for variable sticker counts
5. Generate 4x4 training data, benchmark solver speed

### Phase 3: Scale to 7x7
1. Joint multi-size training (2x2+3x3+4x4+...)
2. May need D=16+ for capacity
3. Consider RL fine-tuning if imitation learning plateaus
4. Test generalization to unseen sizes

## Hardware

- Lambda A100-SXM4-40GB, 30 CPU cores, 221GB RAM
- System Python 3.12, torch 2.7.0
- dwalton solver at /home/ubuntu/rubiks-cube-NxNxN-solver
