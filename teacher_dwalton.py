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
    _BASIC_MOVES = {
        "U", "U'", "U2",
        "R", "R'", "R2",
        "F", "F'", "F2",
        "D", "D'", "D2",
        "L", "L'", "L2",
        "B", "B'", "B2",
    }
    _WIDE_MOVES = {
        "Uw", "Uw'", "Uw2",
        "Rw", "Rw'", "Rw2",
        "Fw", "Fw'", "Fw2",
        "Dw", "Dw'", "Dw2",
        "Lw", "Lw'", "Lw2",
        "Bw", "Bw'", "Bw2",
    }

    if step in _BASIC_MOVES:
        face = step[0]
        if step.endswith("2"):
            turns = 2
        elif step.endswith("'"):
            turns = -1
        else:
            turns = 1
        return Move(face=face, depth=1, width=1, turns=turns)

    if step in _WIDE_MOVES:
        face = step[0]
        suffix = step[2:]  # after "Xw"
        if suffix == "2":
            turns = 2
        elif suffix == "'":
            turns = -1
        else:
            turns = 1
        return Move(face=face, depth=1, width=2, turns=turns)

    raise ValueError(f"Unsupported teacher move for current curriculum: {step}")


def _solve_cube(cube: Cube, solver_class, goal_check) -> tuple[Move, ...]:
    """Generic solver: instantiate solver_class, solve, parse, verify."""
    if goal_check(cube):
        return ()

    _ensure_solver_importable()

    state = cube.to_kociemba_string()
    solver_cube = solver_class(state, "URFDLB", None)
    solver_cube.solve([])
    raw_steps = [s for s in solver_cube.solution if not s.startswith("COMMENT")]
    solution = tuple(_parse_basic_step(step) for step in raw_steps)

    check_cube = cube.copy()
    check_cube.apply_moves(solution)
    if not goal_check(check_cube):
        raise RuntimeError("Teacher solution did not solve cube in local simulator")
    return solution


def solve_cube_222(cube: Cube) -> tuple[Move, ...]:
    if cube.size != 2:
        raise ValueError(f"solve_cube_222 only supports 2x2, got {cube.size}")

    from rubikscubennnsolver.RubiksCube222 import RubiksCube222
    return _solve_cube(cube, RubiksCube222, lambda c: c.has_uniform_faces())


def solve_cube_333(cube: Cube) -> tuple[Move, ...]:
    if cube.size != 3:
        raise ValueError(f"solve_cube_333 only supports 3x3, got {cube.size}")

    from rubikscubennnsolver.RubiksCube333 import RubiksCube333
    return _solve_cube(cube, RubiksCube333, lambda c: c.is_solved())


def solve_cube_444(cube: Cube) -> tuple[Move, ...]:
    if cube.size != 4:
        raise ValueError(f"solve_cube_444 only supports 4x4, got {cube.size}")

    from rubikscubennnsolver.RubiksCube444 import RubiksCube444
    return _solve_cube(cube, RubiksCube444, lambda c: c.is_solved())


def solve_cube_555(cube: Cube) -> tuple[Move, ...]:
    if cube.size != 5:
        raise ValueError(f"solve_cube_555 only supports 5x5, got {cube.size}")

    from rubikscubennnsolver.RubiksCube555 import RubiksCube555
    return _solve_cube(cube, RubiksCube555, lambda c: c.is_solved())
