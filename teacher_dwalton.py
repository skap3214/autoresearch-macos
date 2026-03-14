from __future__ import annotations

import os
import sys
from pathlib import Path

from rubiks import Cube, Move


def _solver_root() -> Path:
    explicit = os.environ.get("DWALTON_SOLVER_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parent.parent / "rubiks-cube-NxNxN-solver"
    if not root.exists():
        raise RuntimeError(
            "dwalton solver repo not found. Clone "
            "https://github.com/dwalton76/rubiks-cube-NxNxN-solver "
            f"to {root} or set DWALTON_SOLVER_ROOT."
        )
    return root


def _ensure_solver_importable():
    root = _solver_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _parse_basic_step(step: str) -> Move:
    if step not in {
        "U", "U'", "U2",
        "R", "R'", "R2",
        "F", "F'", "F2",
        "D", "D'", "D2",
        "L", "L'", "L2",
        "B", "B'", "B2",
    }:
        raise ValueError(f"Unsupported teacher move for current curriculum: {step}")

    face = step[0]
    if step.endswith("2"):
        turns = 2
    elif step.endswith("'"):
        turns = -1
    else:
        turns = 1
    return Move(face=face, depth=1, width=1, turns=turns)


def solve_cube_222(cube: Cube) -> tuple[Move, ...]:
    if cube.size != 2:
        raise ValueError(f"solve_cube_222 only supports 2x2, got {cube.size}")

    # Short-circuit: already solved cubes hang the dwalton solver
    if cube.has_uniform_faces():
        return ()

    _ensure_solver_importable()
    from rubikscubennnsolver.RubiksCube222 import RubiksCube222

    state = cube.to_kociemba_string()
    solver_cube = RubiksCube222(state, "URFDLB", None)
    solver_cube.solve([])
    solution = tuple(_parse_basic_step(step) for step in solver_cube.solution)

    # Verify the move trace solves our simulator state before using it as a label.
    check_cube = cube.copy()
    check_cube.apply_moves(solution)
    if not check_cube.has_uniform_faces():
        raise RuntimeError("Teacher solution did not solve cube in local simulator")
    return solution
