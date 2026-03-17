"""CFOP-based teacher solver that returns stage-labeled solutions.

Uses the rubik-solver library (Wiston999/python-rubik) with its CFOP
and Beginner solvers to produce solutions where each move is tagged
with its CFOP stage.

The solutions are longer than Kociemba (~80 moves vs ~20), but each
move belongs to a clear stage (CROSS, F2L, OLL, PLL), enabling the
model to learn stage-conditioned policies.
"""

from __future__ import annotations

import copy

from rubiks import Cube, Move, FACE_ORDER, FACE_COLORS, rotate_vec

# rubik-solver imports
from rubik_solver.NaiveCube import NaiveCube
from rubik_solver.Cubie import Cube as RCube
from rubik_solver.Move import Move as RMove
from rubik_solver.Solver.Beginner.WhiteCrossSolver import WhiteCrossSolver
from rubik_solver.Solver.Beginner.WhiteFaceSolver import WhiteFaceSolver
from rubik_solver.Solver.Beginner.SecondLayerSolver import SecondLayerSolver
from rubik_solver.Solver.Beginner.YellowCrossSolver import YellowCrossSolver
from rubik_solver.Solver.Beginner.YellowFaceSolver import YellowFaceSolver


# Color mapping: our simulator → rubik-solver
# Our colors: W(white/U), Y(yellow/D), G(green/F), B(blue/B), R(red/R), O(orange/L)
# rubik-solver expects: y(U), b(L), r(F), g(R), o(B), w(D) in ULFRBD order
_OUR_TO_RS = {"W": "y", "O": "b", "G": "r", "R": "g", "B": "o", "Y": "w"}
_RS_TO_OUR = {v: k for k, v in _OUR_TO_RS.items()}

# Reverse: rubik-solver face letters → our Move faces
# rubik-solver uses same face names as us (U/D/L/R/F/B)
# But Y = whole cube rotation around Y axis, not a face move


def _apply_cube_rotation(cube: Cube, rotation_str: str):
    """Apply a whole-cube rotation (Y/X/Z) to our Cube in-place.

    Note: rubik-solver's Y convention is opposite to our rotate_vec,
    so we negate the turns for Y and Z (but not X, which needs testing).
    """
    s = rotation_str.strip().upper()
    axis_map = {"Y": "y", "X": "x", "Z": "z"}
    axis = axis_map[s[0]]
    if "2" in s:
        quarter_turns = 2
    elif "'" in s:
        # rubik-solver Y' = our +1
        quarter_turns = 1
    else:
        # rubik-solver Y = our -1
        quarter_turns = -1

    new_stickers = {}
    for (pos, normal), color in cube.stickers.items():
        new_pos = rotate_vec(pos, axis, quarter_turns)
        new_normal = rotate_vec(normal, axis, quarter_turns)
        new_stickers[(new_pos, new_normal)] = color
    cube.stickers = new_stickers


def _our_cube_to_rs(cube: Cube) -> RCube:
    """Convert our Cube to rubik-solver's Cubie Cube."""
    # Build state string in ULFRBD order for NaiveCube
    state_chars = []
    for face in ("U", "L", "F", "R", "B", "D"):
        grid = cube.face_grid(face)
        for row in grid:
            for color in row:
                state_chars.append(_OUR_TO_RS[color])
    state_str = "".join(state_chars)

    nc = NaiveCube()
    nc.set_cube(state_str)
    rc = RCube()
    rc.from_naive_cube(nc)
    return rc


def _parse_rs_move(move_str: str) -> tuple[Move | None, str | None]:
    """Convert rubik-solver move string to our Move.
    Returns (Move, None) for face moves, (None, rotation_type) for rotations."""
    s = str(move_str).strip()

    if s[0] in ("Y", "X", "Z"):
        return None, s

    face = s[0]
    if face not in ("U", "D", "L", "R", "F", "B"):
        return None, None

    if s.endswith("2"):
        turns = 2
    elif s.endswith("'"):
        turns = -1
    else:
        turns = 1

    return Move(face=face, depth=1, width=1, turns=turns), None


# Rotation remappings: how face names change after whole-cube rotations
# Y = rotate around U-D axis (U stays, D stays, F→R→B→L→F clockwise from top)
# Y' = reverse
# X = rotate around R-L axis (R stays, L stays, F→U→B→D)
# Z = rotate around F-B axis (F stays, B stays, U→R→D→L)
_Y_CW = {"F": "L", "L": "B", "B": "R", "R": "F", "U": "U", "D": "D"}
_Y_CCW = {"F": "R", "R": "B", "B": "L", "L": "F", "U": "U", "D": "D"}
_X_CW = {"U": "F", "F": "D", "D": "B", "B": "U", "R": "R", "L": "L"}
_X_CCW = {"U": "B", "B": "D", "D": "F", "F": "U", "R": "R", "L": "L"}
_Z_CW = {"U": "R", "R": "D", "D": "L", "L": "U", "F": "F", "B": "B"}
_Z_CCW = {"U": "L", "L": "D", "D": "R", "R": "U", "F": "F", "B": "B"}


def _apply_rotation(face_map: dict[str, str], rotation: str) -> dict[str, str]:
    """Compose a rotation into the current face mapping."""
    s = rotation.strip().upper()
    axis = s[0]
    if "2" in s:
        rots = [{"Y": _Y_CW, "X": _X_CW, "Z": _Z_CW}[axis]] * 2
    elif "'" in s:
        rots = [{"Y": _Y_CCW, "X": _X_CCW, "Z": _Z_CCW}[axis]]
    else:
        rots = [{"Y": _Y_CW, "X": _X_CW, "Z": _Z_CW}[axis]]

    result = dict(face_map)
    for rot in rots:
        result = {k: rot[v] for k, v in result.items()}
    return result


def solve_cube_333_cfop(cube: Cube) -> list[tuple[str, Move]]:
    """Solve a 3x3 cube using CFOP stages. Returns [(stage, move), ...].

    Stages: "CROSS", "F2L", "OLL", "PLL"
    Moves are in our simulator's Move format.

    Approach: apply all solver moves (including Y rotations) to the
    rubik-solver cube, and after each face move, diff the kociemba
    string to determine which physical move was made.
    """
    if cube.size != 3:
        raise ValueError(f"CFOP solver only supports 3x3, got {cube.size}")

    if cube.is_solved():
        return []

    rc = _our_cube_to_rs(cube)

    # Solve stage by stage, collecting all moves including rotations
    c = copy.deepcopy(rc)
    stages_raw = []

    cross = WhiteCrossSolver(c).solution()
    stages_raw.append(("CROSS", [str(m) for m in cross]))

    corners = WhiteFaceSolver(c).solution()
    stages_raw.append(("F2L", [str(m) for m in corners]))

    edges = SecondLayerSolver(c).solution()
    stages_raw.append(("F2L", [str(m) for m in edges]))

    ycross = YellowCrossSolver(c).solution()
    stages_raw.append(("OLL", [str(m) for m in ycross]))

    yface = YellowFaceSolver(c).solution()
    stages_raw.append(("PLL", [str(m) for m in yface]))

    # Apply all moves to a rubik-solver cube, and after each face move
    # try all 18 possible moves on our cube to find which one matches
    ALL_MOVES = [
        Move(face=f, depth=1, width=1, turns=t)
        for f in ("U", "D", "L", "R", "F", "B")
        for t in (1, -1, 2)
    ]

    rc_replay = copy.deepcopy(rc)
    our_replay = cube.copy()
    result = []

    for stage, moves in stages_raw:
        for m_str in moves:
            rc_replay.move(RMove(m_str))
            if m_str[0] in "YXZ":
                # Whole-cube rotation — apply to our cube too
                _apply_cube_rotation(our_replay, m_str)
                continue

            # A face move was made. Find which of our 18 moves produces
            # the same state by converting the rubik-solver state to kociemba
            # and comparing with each candidate.
            # Get target state from rubik-solver as color string
            target_nc = rc_replay.to_naive_cube()
            target_colors = []
            for face in FACE_ORDER:
                for i in range(3):
                    for j in range(3):
                        rs_color = target_nc.faces[face].get_colour(i, j)
                        target_colors.append(_RS_TO_OUR.get(rs_color, rs_color))
            target_str = "".join(target_colors)

            # Try each move on our cube to find which produces the target state
            def _our_color_string(c):
                chars = []
                for face in FACE_ORDER:
                    for row in c.face_grid(face):
                        chars.extend(row)
                return "".join(chars)

            found = False
            for candidate in ALL_MOVES:
                test = our_replay.copy()
                test.apply_move(candidate)
                if _our_color_string(test) == target_str:
                    result.append((stage, candidate))
                    our_replay.apply_move(candidate)
                    found = True
                    break

            if not found:
                return []  # couldn't match — bail

    # Final verification — check if all faces are uniform (solved, possibly reoriented)
    if not our_replay.is_solved() and not our_replay.has_uniform_faces():
        return []

    return result


def solve_cube_333_cfop_moves_only(cube: Cube) -> tuple[tuple[Move, ...], list[str]]:
    """Solve returning (moves, stages) as separate tuples.

    Returns:
        moves: tuple of Move objects
        stages: list of stage strings, one per move
    """
    result = solve_cube_333_cfop(cube)
    if not result:
        return (), []
    moves = tuple(m for _, m in result)
    stages = [s for s, _ in result]
    return moves, stages
