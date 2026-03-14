# Rubik's Solver Methodology And Roadmap v2

This document consolidates the current direction for training a model to solve Rubik's cubes using the `autoresearch-macos` repository as the experimentation harness.

It captures:
- what problem we are actually trying to solve,
- how we want to represent states and moves,
- how generalization should be tested,
- what the teacher/oracle is,
- what the training curriculum should be,
- how the autonomous optimization loop should eventually be run.

## Project goal

The long-term goal is not merely to imitate scramble inversions.

The real goal is:

- train a model that can solve cubes from state input,
- begin with `2x2`,
- then progressively extend to `3x3`, `4x4`, `5x5`,
- and later test whether the learned representation and policy generalize to larger unseen cube sizes such as `6x6` through `9x9`.

This is a learning/generalization problem, not a "does a solver exist?" problem.

## What is already solved vs what is not

Algorithmic `NxN` cube solving is already solved in practice.
Existing hand-engineered solvers can solve large cubes reliably.

What is still interesting here is:
- whether a learned model can absorb that solving behavior,
- whether it can operate from a canonicalized state representation,
- whether it can scale across cube sizes,
- whether it can generalize beyond the sizes seen during training.

So the existing algorithmic solver is a teacher, not the thing we are trying to replace conceptually on day one.

## Core methodology

### 1. Use a real solver as the teacher

We should use `dwalton76/rubiks-cube-NxNxN-solver` as the main teacher/oracle.

Reason:
- it already solves `NxN` cubes,
- it gives us a principled source of solve traces,
- it is stronger and cleaner than supervising on inverse-scramble sequences,
- it turns the project into imitation of a real solver rather than imitation of the scramble generator.

For the first milestone, we are using the `2x2` path from the `dwalton` ecosystem.

### 2. Use canonicalized cube state as input

The model should not take scramble strings as its primary input.

The model input should be:
- the current cube state,
- serialized in a fixed global frame,
- with a stable face orientation convention.

Canonical frame:
- `U = white`
- `F = green`

Why this matters:
- the state is the real problem instance,
- many scramble strings can map to equivalent states,
- canonicalization reduces needless variance,
- it makes the learning problem more about solving and less about textual inversion.

### 3. Use structured move tokenization

Do not use generic BPE over raw move strings like `3Rw2`.

Moves should be represented compositionally.

Example:

```text
<MOVE>
<FACE> FACE_R </FACE>
<DEPTH> DIGIT_3 </DEPTH>
<WIDTH> DIGIT_2 </WIDTH>
<TURN> TURN_HALF </TURN>
</MOVE>
```

Why:
- it avoids accidental size-specific tokens,
- it keeps the vocabulary stable across cube sizes,
- it makes extension beyond single-digit layers possible,
- it supports later `NxN` scaling better than raw notation fragments.

### 4. Start with next-move prediction

The first model should predict the next move given:
- cube size,
- current cube state,
- later possibly stage metadata or short action history.

We should not begin with full-sequence generation as the primary training target.

Why:
- many valid full solutions exist,
- next-step prediction is easier to supervise,
- it lets us evaluate through rollouts in the simulator,
- it is the most natural first milestone for a teacher-imitation setup.

## Why inverse-scramble supervision is not enough

We previously considered training on the inverse of the scramble.

That was acceptable as a basic systems baseline, but it is not the right end state.

Problems with inverse-scramble supervision:
- it teaches the model to undo one particular history, not to solve from state,
- the same state can have different labels depending on how it was generated,
- it does not reflect real solving strategy,
- it encourages narrow recovery behavior instead of reusable solving abstractions.

Using a solver-generated trace is a much better form of supervision because the labels are tied to actual solution behavior rather than to scramble history.

## Current staged curriculum

The training plan should be staged, not flat.

### Stage A: `2x2` only

First checkpoint:

Can the model solve held-out `2x2` cubes from state input?

This is the first real milestone because:
- `2x2` is the simplest cube domain,
- solve traces are shortest,
- debugging is fastest,
- rollout failures are easier to inspect.

### Stage B: `3x3`

After `2x2` works:
- add `3x3` teacher traces,
- preserve held-out `2x2` validation,
- verify that the same setup scales to the standard cube.

### Stage C: `4x4` and `5x5`

After `3x3` is working:
- add `4x4`, then `5x5`,
- keep per-size held-out validation,
- verify that the move representation still scales cleanly,
- later consider stage-conditioned supervision if reduction traces become important.

### Stage D: OOD evaluation

Only after training on `2x2` through `5x5` should we ask the real generalization question using:
- `6x6`
- `7x7`
- `8x8`
- `9x9`

This is when we evaluate whether size-general behavior emerges.

## Generalization framing

Generalization should be tested explicitly, not inferred from one blended metric.

Recommended split structure:

- in-distribution train: sizes seen during training
- in-distribution validation: held-out states from those same sizes
- out-of-distribution evaluation: larger unseen cube sizes

For the later full experiment:
- train: `2x2` through `5x5`
- ID validation: held-out `2x2` through `5x5`
- OOD evaluation: `6x6` through `9x9`

But for the current milestone:
- train: `2x2`
- validation: held-out `2x2`

That is enough to answer the first question:
- can the model learn to solve the smallest cube at all?

## Evaluation methodology

Do not use exact sequence match as the main success criterion.

Why:
- multiple move sequences can be valid,
- sequence equality is much weaker than actual solving ability,
- the model should be judged by what happens when its actions are executed.

Primary metrics should come from simulator rollouts.

Recommended metrics:
- `solve_rate@K`
- `id_move_accuracy`
- `valid_move_rate`
- `mean_residual_error`
- `mean_steps_to_solve`

### Current milestone metric

For the current `2x2` stage, the primary metric should be:
- held-out `2x2 solve_rate`

Token-level loss matters for debugging, but it is not the final objective.

## Important caveat for `2x2`

`2x2` cubes have no fixed centers.

That means:
- a teacher solver may return a solution that solves the cube up to a whole-cube rotation,
- the cube can be "solved" in a structural sense without matching one fixed canonical face-color orientation.

So for the `2x2` milestone, the goal check should be:
- all faces are uniform,

not necessarily:
- exact canonical solved color layout.

This is a subtle but important evaluation detail.

## Tokenization strategy

The tokenization strategy should remain stable as we scale up.

### State tokens

Use face-grid serialization with fixed face ordering:
- `U`
- `R`
- `F`
- `D`
- `L`
- `B`

Example shape:

```text
<TASK_POLICY>
<SIZE> DIGIT_5 </SIZE>
<STATE>
<GRID_U>
<ROW> COL_W COL_W COL_W COL_W COL_W </ROW>
...
</GRID_U>
...
</STATE>
<TARGET>
```

Why this works:
- vocabulary stays fixed,
- larger cubes become longer sequences rather than introducing arbitrary new symbol types,
- the representation stays aligned with the canonical frame.

### Move tokens

Represent moves with explicit structure:
- face,
- depth,
- width,
- turn.

Numeric fields should use digit tokens, not one token per entire number.

This preserves compatibility with future multi-digit layer indices.

## Training loop role of this repo

This repo is a harness, not the intelligence itself.

Its role is to provide:
- the training loop,
- local evaluation,
- experiment logging,
- a compact code surface for iterative improvement.

The actual "autonomous loop" is:
- update code,
- run training,
- read the metric,
- keep or discard the change,
- repeat.

For this project, that loop should eventually optimize solve-rate metrics, not language-model `val_bpb`.

## Current implementation state

At this point, the repo has already been adapted to include:
- a custom `NxN` simulator,
- structured move tokens,
- canonical face-grid state serialization,
- a browser playground for manual inspection,
- a `2x2` teacher wrapper around `dwalton76`,
- local run logging.

What that means:
- the infrastructure exists,
- the teacher-backed `2x2` training setup exists,
- the next job is improving policy quality rather than building the entire pipeline from scratch.

## Why the first full `2x2` run still failed to solve held-out cubes

The first teacher-backed `2x2` run showed:
- training loss dropped sharply,
- move imitation improved,
- held-out rollout solve rate was still `0`.

This suggests:
- the model is learning local token prediction,
- but the rollout policy is still weak,
- and the gap is now in policy execution quality, not in basic data loading.

That is still a useful milestone because it means the system is now failing in the interesting place.

## Most obvious next improvements

The next improvements should not be random architecture fiddling.

The highest-value next steps are:

### 1. Constrained decoding

At generation time, force the model to emit only legal token continuations and legal moves for the current cube size.

Why:
- reduces invalid outputs,
- shrinks the search space,
- makes rollouts more stable.

### 2. Add short action history to the prompt

Give the model the previous few moves.

Why:
- reduces oscillation,
- helps avoid immediate undo loops,
- makes policy rollouts less myopic.

### 3. Add rollout-time search or heuristics

Examples:
- beam search,
- repeated-state penalties,
- no-immediate-inverse rules,
- simple cycle avoidance.

Why:
- greedy decoding is usually too weak for combinatorial control tasks.

### 4. Inspect failure modes directly

For held-out `2x2` examples:
- record model rollouts,
- see whether it emits valid moves,
- see whether it loops,
- see whether it nearly solves but fails late.

This is likely more useful right now than changing the transformer architecture.

## Recommended near-term roadmap

### Milestone 1

Get nonzero held-out `2x2` solve rate.

Work required:
- teacher-backed data,
- constrained decoding,
- rollout stabilization,
- inspection of failures.

### Milestone 2

Push `2x2` solve rate to something convincingly nontrivial.

Work required:
- tune prompt structure,
- tune rollout logic,
- possibly tune model size or sequence formatting.

### Milestone 3

Add `3x3`.

Only after `2x2` is genuinely working should we expand the curriculum.

### Milestone 4

Add `4x4` and `5x5`.

At that point, begin thinking about stage-conditioned traces if reduction-style teaching is necessary.

### Milestone 5

Evaluate on OOD larger cubes.

That is where the actual generalization question becomes meaningful.

## Final takeaway

The methodology is now:

- use a real `NxN` solver as the teacher,
- train from canonicalized cube states,
- use structured move tokenization,
- start with next-move prediction,
- validate with simulator rollouts,
- solve `2x2` first,
- only then scale up and ask the size-generalization question.

That is the clearest and most defensible path toward the research goal.
