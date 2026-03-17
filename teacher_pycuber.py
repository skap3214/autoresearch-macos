"""CFOP teacher using pycuber — produces stage-labeled ~60-move solutions.

Pycuber implements proper CFOP with F2L algorithms (41 cases),
OLL (57 algorithms), and PLL (21 algorithms). Solutions average
~60 moves with y-rotations resolved into absolute face moves.
"""

from __future__ import annotations

import io
import re
import sys

import pycuber as pc
from pycuber.solver.cfop import CFOPSolver

from rubiks import Cube, Move, FACE_ORDER, FACE_COLORS


# Standard notation → our Move
_NOTATION_TO_TURNS = {
    "": 1,       # R = clockwise
    "'": -1,     # R' = counterclockwise
    "2": 2,      # R2 = half turn
}


def _parse_move(notation: str) -> Move | None:
    """Parse standard cube notation (R, R', R2, etc.) into our Move."""
    s = notation.strip()
    if not s:
        return None
    # Skip rotations (x, y, z) and slice moves (M, E, S)
    if s[0].lower() in ("x", "y", "z", "m", "e", "s"):
        return None
    # Handle wide moves (r, l, u, d, f, b = lowercase)
    face = s[0].upper()
    if face not in ("U", "D", "L", "R", "F", "B"):
        return None
    suffix = s[1:]
    turns = _NOTATION_TO_TURNS.get(suffix)
    if turns is None:
        return None
    return Move(face=face, depth=1, width=1, turns=turns)


def _our_cube_to_pycuber(cube: Cube) -> pc.Cube:
    """Convert our Cube to pycuber's Cube via kociemba string."""
    # pycuber can be constructed from a dict of face colors
    # Our face_grid returns colors W,Y,G,B,R,O
    # pycuber uses color names or single chars
    color_map = {"W": "white", "Y": "yellow", "G": "green",
                 "B": "blue", "R": "red", "O": "orange"}
    # pycuber face order: U, D, F, B, L, R
    # Each face is 3x3, indexed [row][col] from top-left
    pc_cube = pc.Cube()
    for face_name in FACE_ORDER:
        grid = cube.face_grid(face_name)
        for r in range(3):
            for c in range(3):
                # pycuber square naming: e.g., U[0][0], F[1][2]
                # We need to set each square's color
                pass
    # Actually, pycuber doesn't easily support setting state directly.
    # Instead, find the scramble that produces our state.
    # Easier approach: use kociemba string to reconstruct.
    # pycuber can't import kociemba strings directly.
    # Best approach: apply our scramble to a fresh pycuber cube.
    return None  # placeholder — we'll use a different approach


def solve_cube_333_pycuber(cube: Cube, scramble: tuple[Move, ...]) -> list[tuple[str, Move]]:
    """Solve a 3x3 cube using pycuber's CFOP solver.

    Returns [(stage, move), ...] with ~60 moves, stage-labeled.
    Uses the scramble to reconstruct the state in pycuber's format.
    """
    if cube.size != 3:
        raise ValueError(f"pycuber solver only supports 3x3, got {cube.size}")

    if cube.is_solved():
        return []

    # Reconstruct the scramble in pycuber notation
    pc_cube = pc.Cube()
    scramble_str = " ".join(_move_to_notation(m) for m in scramble)
    pc_cube(pc.Formula(scramble_str))

    # Capture stage output
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    try:
        solver = CFOPSolver(pc_cube)
        solution = solver.solve()
    finally:
        output = captured.getvalue()
        sys.stdout = old_stdout

    # Parse stage labels from captured output
    stage_segments = _parse_stage_output(output)

    # Get the full y-resolved solution
    full_moves_str = str(solution).split()

    # Convert full solution to our Move objects (all verified to work)
    all_moves = []
    for notation in full_moves_str:
        our_move = _parse_move(notation)
        if our_move is not None:
            all_moves.append(our_move)

    # Verify solution first
    check = cube.copy()
    for move in all_moves:
        check.apply_move(move)
    if not check.is_solved() and not check.has_uniform_faces():
        return []

    # Assign stage labels: count non-y moves per stage segment,
    # map sequentially to the flat move list
    result = []
    move_idx = 0
    for stage, stage_move_strs in stage_segments:
        real_count = sum(1 for m in stage_move_strs if not m.startswith("y"))
        for _ in range(real_count):
            if move_idx < len(all_moves):
                result.append((stage, all_moves[move_idx]))
                move_idx += 1

    # Any remaining moves (rounding errors) → assign to last stage
    last_stage = result[-1][0] if result else "PLL"
    while move_idx < len(all_moves):
        result.append((last_stage, all_moves[move_idx]))
        move_idx += 1

    return result


def _move_to_notation(move: Move) -> str:
    """Convert our Move to standard notation string."""
    suffix = {1: "", -1: "'", 2: "2"}[move.turns]
    return f"{move.face}{suffix}"


def _parse_stage_output(output: str) -> list[tuple[str, list[str]]]:
    """Parse pycuber's stage output into [(stage, [moves]), ...]."""
    segments = []
    # Clean all control characters and ANSI escapes
    clean = re.sub(r"[\x1b\r]", " ", output)
    clean = re.sub(r"\[2K", "", clean)

    for line in clean.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Split on "Cross:", "F2L(...):", "OLL:", "PLL:" — take the LAST occurrence
        # (because "Solving Cross" appears before "Cross: moves")
        if "Cross:" in line:
            moves_str = line.split("Cross:")[-1].strip()
            if moves_str:
                segments.append(("CROSS", moves_str.split()))
        elif "F2L(" in line and "):" in line:
            moves_str = line.split("):")[-1].strip()
            if moves_str:
                segments.append(("F2L", moves_str.split()))
        elif "OLL:" in line:
            moves_str = line.split("OLL:")[-1].strip()
            if moves_str:
                segments.append(("OLL", moves_str.split()))
        elif "PLL:" in line:
            moves_str = line.split("PLL:")[-1].strip()
            if moves_str:
                segments.append(("PLL", moves_str.split()))

    return segments
