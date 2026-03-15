# Training Report: 2x2 Rubik's Cube Solver via Imitation Learning

## Key Facts

| Property | Value |
|---|---|
| Model | Transformer (GPT-style), D=8, dim=512, 8 heads, 25.4M params |
| Task | 2x2 Rubik's Cube solving via imitation learning |
| Input | 24 sticker colors (flat encoding) + last 3 moves as history |
| Output | Single token from 19 classes (18 MOVE\_face\_turn + DONE) |
| Training | Supervised (cross-entropy) + auxiliary value head (MSE, weight 0.5) predicting distance-to-goal |
| DAgger | Mid-training on-policy collection at 50% of training: rollout on 200 random cubes, query teacher for corrections, adds ~2500 on-policy examples |
| Teacher | dwalton76/rubiks-cube-NxNxN-solver (optimal solver) |
| Evaluation | Rollout on 256 held-out scrambled cubes with hybrid search (model score + residual heuristic + state avoidance + no-inverse rule) |

## Results Progression

### Phase 1: Early Experiments (0% solve rate)

Structured token formats and unconstrained or constrained decoding. The model learned move accuracy but could not solve any cubes end-to-end.

| Experiment | Description | Move Acc | Solve Rate | Notes |
|---|---|---|---|---|
| 140918 | D=4 dim=256, structured 14-token, unconstrained | 21.8% | 0/256 | Baseline |
| 142741 | Constrained decoding, 256 episodes | 33.7% | 0/256 | |
| 143626 | +4096 episodes (38K examples) | 37.0% | 0/256 | |
| 144956 | Compact 3-token answer format | 43.9% | 0/256 | |
| 150805 | Flat state (no XML markers) | 44.2% | 0/256 | |

### Phase 2: Key Breakthroughs

| Experiment | Description | Solve Rate | Key Insight |
|---|---|---|---|
| exp6 | +action history(3) +no-inverse rule | 4/256 (1.6%) | First solves ever |
| exp10 | Joint MOVE\_face\_turn tokens (19-class) | 0/256 | Higher move acc (53%) but no solves without search |
| exp13 | Hybrid search + 2048 episodes | 12/256 (4.7%) | Search unlocks generalization |
| exp15 | 4096 episodes + hybrid search | 20/256 (7.8%) | More data helps |
| dagger1 | DAgger mid-training (2995 on-policy examples) | 40/256 (15.6%) | Biggest single lever |
| best | DAgger + auxiliary value head + residual search | 41/256 (16.0%) | Combined best techniques |
| rs100 | ROLLOUT\_MIN\_STEPS=100 | 56/256 (21.9%) | Model needed more eval steps |

### Phase 3: Scaling on MPS (MacBook)

| Experiment | Episodes | Time | Solve Rate | Notes |
|---|---|---|---|---|
| t1200e8k | 8192 | 20 min | 103/256 (40.2%) | More data + time wins |

### Phase 4: GPU Scaling (RTX 5090)

| Experiment | Model | Episodes | Time | Solve Rate | Notes |
|---|---|---|---|---|---|
| d8gpu | D=8, 25.4M params | 32K | 60 min | 239/256 (93.4%) | Larger model + more data |
| d8e64k | D=8, 25.4M params | 64K | 60 min | 256/256 (100%) | +ROLLOUT\_MIN\_STEPS=200 |

## What Worked

| Technique | Evidence | Impact |
|---|---|---|
| DAgger (mid-training on-policy data) | 20/256 -> 40/256 | ~2x, biggest single lever |
| Auxiliary value loss (weight 0.5) | 41/256 vs 9/256 ablation (noval) | 3.5x multiplier |
| Joint MOVE tokens (single token prediction) | Move acc 44% -> 53% | Simplified output space |
| Flat state encoding (no XML markers) | Shorter sequences, same accuracy | Faster training |
| Hybrid search (residual delta=2) | Unlocked first solves beyond greedy | Essential for generalization |
| Scaling data + compute together | 4K eps/10min -> 8K/20min -> 64K/60min | Consistent gains at every scale |
| Larger model on GPU (D=8, 25.4M params) | 40.2% -> 93.4% | Capacity was the bottleneck |
| ROLLOUT\_MIN\_STEPS=200 | 93.4% -> 100% | Model could solve more given time |

## What Didn't Work

| Technique | Experiment | Result |
|---|---|---|
| Bigger models on MPS | exp 150142 (D=6) | Too slow, fewer epochs, no gain |
| Value-guided search on MPS | valsrch, mlpsrch | Per-call overhead made eval take >1h |
| Value-primary loss (weight > 0.5) | vprimr (0.2 CE + 1.0 MSE) | Policy degraded, solve rate dropped |
| More DAgger data or rounds | dagger2a (300 eps), 2rnd (two rounds) | Over-diluted training distribution |
| Curriculum learning | curr1, curr3 vs nocurr | No clear benefit over uniform sampling |
| Weight decay | wd01 (0.1) | Regularization hurt, dropped to 11/256 |
| SEARCH\_RESIDUAL\_DELTA=1 | rd1 | 0/256 -- too restrictive, blocks sacrifice moves |
| SEARCH\_RESIDUAL\_DELTA=3 | rd3 | No gain over delta=2, too permissive |
| Longer training on same data | t1200 (20min, 4K eps) | Overfitted (loss=0.27 but no generalization gain) |

## Architecture Details

| Component | Value |
|---|---|
| Architecture | GPT-style Transformer |
| Depth | 8 layers |
| Model dim | 512 |
| Heads | 8 (GQA) |
| Parameters | 25.4M |
| Optimizer | MuonAdamW (Muon for matrix params, AdamW for embeddings/scalars) |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Value Embeddings | ResFormer-style |
| Logit capping | Soft-capping at 15 |
| Compilation | torch.compile enabled on CUDA |

## Final Model Stats

| Metric | Value |
|---|---|
| Parameters | 25.4M |
| Training episodes | 64K (615K training examples + ~2500 DAgger examples) |
| Training steps | 51,547 |
| Training time | 60 minutes |
| Solve rate | 100% (256/256 held-out cubes) |
| Peak VRAM | 3.2 GB (RTX 5090) |
| MFU | 4.1% |
| Hardware | NVIDIA RTX 5090 |

## Full Experiment Log

All 28 experiments from `results.tsv`, ordered chronologically:

| # | Experiment | Solve Rate | Move Acc | Mean Residual | Steps | Status | Description |
|---|---|---|---|---|---|---|---|
| 1 | 140918 | 0.0% | 21.8% | 20.25 | 247 | keep | Baseline: D=4 dim=256, structured 14-tok, unconstrained |
| 2 | 142741 | 0.0% | 33.7% | 20.33 | 3596 | keep | Constrained decoding, 256 episodes |
| 3 | 143626 | 0.0% | 37.0% | 20.00 | 3632 | keep | +4096 episodes (38K examples) |
| 4 | 144956 | 0.0% | 43.9% | 19.95 | 5128 | keep | Compact 3-token format |
| 5 | 150142 | 0.0% | 34.9% | 19.92 | 3259 | discard | D=6 dim=384 10.7M -- too slow |
| 6 | 150805 | 0.0% | 44.2% | 19.48 | 4986 | keep | Flat state (no markers) |
| 7 | exp6 | 1.6% | 40.2% | 19.50 | 5326 | keep | +action history + no-inverse rule |
| 8 | exp10 | 0.0% | 53.0% | -- | 5913 | keep | Joint MOVE tokens (19-class) |
| 9 | exp13 | 4.7% | 52.0% | 10.62 | 6158 | keep | Hybrid search + 2048 episodes |
| 10 | exp15 | 7.8% | 50.0% | 10.52 | 4774 | keep | 4096 episodes + hybrid search |
| 11 | dagger1 | 15.6% | 61.5% | 9.15 | 5410 | keep | DAgger mid-training (2995 on-policy examples) |
| 12 | valaux | 11.3% | 57.1% | 9.54 | 4879 | keep | Value head as auxiliary loss |
| 13 | best | 16.0% | 62.6% | 8.97 | 5196 | keep | DAgger + aux value + residual search |
| 14 | lr10 | 16.0% | 63.0% | 8.21 | 5004 | keep | MATRIX\_LR=0.10 (best mean residual) |
| 15 | rs100 | 21.9% | 61.4% | 7.80 | 4158 | keep | ROLLOUT\_MIN\_STEPS=100 |
| 16 | t1200e8k | 40.2% | 67.9% | 5.95 | 19660 | keep | 8K eps + 20min (MPS scaling) |
| 17 | d8gpu | 93.4% | 82.6% | -- | 50078 | keep | D=8 + 32K eps + 60min (RTX 5090) |
| 18 | d8e64k | 100.0% | 84.0% | -- | 51547 | keep | D=8 + 64K eps + ROLLOUT\_MIN\_STEPS=200 |
