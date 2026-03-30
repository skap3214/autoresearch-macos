from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable


FACE_ORDER = ("U", "R", "F", "D", "L", "B")
FACE_NORMALS = {
    "U": (0, 1, 0),
    "D": (0, -1, 0),
    "F": (0, 0, 1),
    "B": (0, 0, -1),
    "R": (1, 0, 0),
    "L": (-1, 0, 0),
}
FACE_COLORS = {
    "U": "W",
    "D": "Y",
    "F": "G",
    "B": "B",
    "R": "R",
    "L": "O",
}
FACE_AXES = {
    "U": ("y", 1),
    "D": ("y", -1),
    "F": ("z", 1),
    "B": ("z", -1),
    "R": ("x", 1),
    "L": ("x", -1),
}
FACE_CW_QUARTER_TURNS = {
    "U": -1,
    "D": 1,
    "R": -1,
    "L": 1,
    "F": -1,
    "B": 1,
}

# Face-local basis vectors used to serialize each face in a fixed orientation.
FACE_VIEW_BASIS = {
    "U": ((0, 0, -1), (1, 0, 0)),
    "D": ((0, 0, 1), (1, 0, 0)),
    "F": ((0, 1, 0), (1, 0, 0)),
    "B": ((0, 1, 0), (-1, 0, 0)),
    "R": ((0, 1, 0), (0, 0, -1)),
    "L": ((0, 1, 0), (0, 0, 1)),
}


@dataclass(frozen=True)
class Move:
    face: str
    depth: int = 1
    width: int = 1
    turns: int = 1

    def __post_init__(self):
        if self.face not in FACE_ORDER:
            raise ValueError(f"Invalid face: {self.face}")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        if self.width < 1:
            raise ValueError(f"width must be >= 1, got {self.width}")
        if self.turns not in (-1, 1, 2):
            raise ValueError(f"turns must be one of -1, 1, 2, got {self.turns}")

    def inverse(self) -> "Move":
        if self.turns == 2:
            return self
        return Move(self.face, self.depth, self.width, -self.turns)

    def turn_name(self) -> str:
        return {1: "CW", -1: "CCW", 2: "HALF"}[self.turns]

    def to_dict(self) -> dict[str, int | str]:
        return {
            "face": self.face,
            "depth": self.depth,
            "width": self.width,
            "turns": self.turns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int | str]) -> "Move":
        return cls(
            face=str(data["face"]),
            depth=int(data["depth"]),
            width=int(data["width"]),
            turns=int(data["turns"]),
        )

    def __str__(self) -> str:
        return (
            f"Move(face={self.face}, depth={self.depth}, "
            f"width={self.width}, turns={self.turn_name()})"
        )


@dataclass(frozen=True)
class Episode:
    size: int
    scramble: tuple[Move, ...]
    solution: tuple[Move, ...]
    max_rollout_steps: int

    def to_dict(self) -> dict[str, object]:
        return {
            "size": self.size,
            "scramble": [move.to_dict() for move in self.scramble],
            "solution": [move.to_dict() for move in self.solution],
            "max_rollout_steps": self.max_rollout_steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Episode":
        return cls(
            size=int(data["size"]),
            scramble=tuple(Move.from_dict(item) for item in data["scramble"]),
            solution=tuple(Move.from_dict(item) for item in data["solution"]),
            max_rollout_steps=int(data["max_rollout_steps"]),
        )


def _dot(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _rotate_once(vec: tuple[int, int, int], axis: str) -> tuple[int, int, int]:
    x, y, z = vec
    if axis == "x":
        return (x, -z, y)
    if axis == "y":
        return (z, y, -x)
    if axis == "z":
        return (-y, x, z)
    raise ValueError(f"Unsupported axis: {axis}")


def rotate_vec(vec: tuple[int, int, int], axis: str, quarter_turns: int) -> tuple[int, int, int]:
    quarter_turns %= 4
    result = vec
    for _ in range(quarter_turns):
        result = _rotate_once(result, axis)
    return result


class Cube:
    def __init__(self, size: int):
        if size < 2:
            raise ValueError(f"Cube size must be >= 2, got {size}")
        self.size = size
        self.limit = size - 1
        self.coords = tuple(range(-self.limit, self.limit + 1, 2))
        self.stickers: dict[tuple[tuple[int, int, int], tuple[int, int, int]], str] = {}
        self._init_solved()

    def _init_solved(self):
        limit = self.limit
        for face, normal in FACE_NORMALS.items():
            color = FACE_COLORS[face]
            nx, ny, nz = normal
            for a in self.coords:
                for b in self.coords:
                    if face == "U":
                        position = (a, limit, b)
                    elif face == "D":
                        position = (a, -limit, b)
                    elif face == "F":
                        position = (a, b, limit)
                    elif face == "B":
                        position = (a, b, -limit)
                    elif face == "R":
                        position = (limit, b, a)
                    elif face == "L":
                        position = (-limit, b, a)
                    else:
                        raise ValueError(face)
                    self.stickers[(position, (nx, ny, nz))] = color

    def copy(self) -> "Cube":
        other = Cube(self.size)
        other.stickers = dict(self.stickers)
        return other

    def is_solved(self) -> bool:
        for face in FACE_ORDER:
            expected = FACE_COLORS[face]
            grid = self.face_grid(face)
            if any(color != expected for row in grid for color in row):
                return False
        return True

    def has_uniform_faces(self) -> bool:
        for face in FACE_ORDER:
            grid = self.face_grid(face)
            first = grid[0][0]
            if any(color != first for row in grid for color in row):
                return False
        return True

    def _layer_values(self, face: str, depth: int, width: int) -> set[int]:
        axis, sign = FACE_AXES[face]
        limit = self.limit
        if sign > 0:
            ordered = list(range(limit, -limit - 1, -2))
        else:
            ordered = list(range(-limit, limit + 1, 2))
        end = depth - 1 + width
        if end > len(ordered):
            raise ValueError(
                f"Move {face} depth={depth} width={width} exceeds cube size {self.size}"
            )
        return set(ordered[depth - 1:end])

    def apply_move(self, move: Move):
        axis, _ = FACE_AXES[move.face]
        base_turns = FACE_CW_QUARTER_TURNS[move.face]
        quarter_turns = base_turns * move.turns
        layer_values = self._layer_values(move.face, move.depth, move.width)

        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        rotated: dict[tuple[tuple[int, int, int], tuple[int, int, int]], str] = {}
        for (position, normal), color in self.stickers.items():
            if position[axis_idx] in layer_values:
                new_position = rotate_vec(position, axis, quarter_turns)
                new_normal = rotate_vec(normal, axis, quarter_turns)
                rotated[(new_position, new_normal)] = color
            else:
                rotated[(position, normal)] = color
        self.stickers = rotated

    def apply_moves(self, moves: Iterable[Move]):
        for move in moves:
            self.apply_move(move)

    def face_grid(self, face: str) -> list[list[str]]:
        normal = FACE_NORMALS[face]
        up_vec, right_vec = FACE_VIEW_BASIS[face]
        grid = [["?" for _ in range(self.size)] for _ in range(self.size)]
        for (position, sticker_normal), color in self.stickers.items():
            if sticker_normal != normal:
                continue
            row_val = -_dot(position, up_vec)
            col_val = _dot(position, right_vec)
            row = (row_val + self.limit) // 2
            col = (col_val + self.limit) // 2
            grid[row][col] = color
        return grid

    def state_tokens(self) -> list[str]:
        tokens = ["<STATE>"]
        for face in FACE_ORDER:
            tokens.append(f"<GRID_{face}>")
            for row in self.face_grid(face):
                tokens.append("<ROW>")
                tokens.extend(f"COL_{color}" for color in row)
                tokens.append("</ROW>")
            tokens.append(f"</GRID_{face}>")
        tokens.append("</STATE>")
        return tokens

    def to_kociemba_string(self) -> str:
        color_to_face = {
            "W": "U",
            "R": "R",
            "G": "F",
            "Y": "D",
            "O": "L",
            "B": "B",
        }
        chars: list[str] = []
        for face in FACE_ORDER:
            for row in self.face_grid(face):
                for color in row:
                    chars.append(color_to_face[color])
        return "".join(chars)


def int_to_digit_tokens(value: int) -> list[str]:
    return [f"DIGIT_{digit}" for digit in str(value)]


def digit_tokens_to_int(tokens: Iterable[str]) -> int:
    digits = []
    for token in tokens:
        if not token.startswith("DIGIT_"):
            raise ValueError(f"Invalid digit token: {token}")
        digits.append(token.split("_", 1)[1])
    if not digits:
        raise ValueError("Expected at least one digit token")
    return int("".join(digits))


def detect_cfop_stage(cube: Cube) -> str:
    """Detect which CFOP stage a 3x3 cube is in, checking from solved backwards.

    Returns one of: "CROSS", "F2L", "OLL", "PLL", "SOLVED".
    Only valid for 3x3. Raises ValueError for other sizes.
    """
    if cube.size != 3:
        raise ValueError(f"CFOP stage detection is only valid for 3x3, got {cube.size}x{cube.size}")

    if cube.is_solved():
        return "SOLVED"

    d_grid = cube.face_grid("D")
    d_center = d_grid[1][1]

    # Check if F2L is complete: D face fully uniform + bottom two rows of
    # F, R, B, L each match their own center color.
    def _f2l_done() -> bool:
        # D face must be fully uniform (all 9 stickers match center)
        for row in d_grid:
            for color in row:
                if color != d_center:
                    return False
        # Bottom two rows of each lateral face must match that face's center
        for face in ("F", "R", "B", "L"):
            grid = cube.face_grid(face)
            center = grid[1][1]
            # Rows are indexed 0=top, 1=middle, 2=bottom for a 3x3.
            # "Bottom two rows" = rows 1 and 2 (the two rows adjacent to D).
            for r in range(1, 3):
                for c in range(3):
                    if grid[r][c] != center:
                        return False
        return True

    f2l_done = _f2l_done()

    if f2l_done:
        # Check OLL: all 9 stickers on U face are the same color
        u_grid = cube.face_grid("U")
        u_center = u_grid[1][1]
        oll_done = all(u_grid[r][c] == u_center for r in range(3) for c in range(3))
        if oll_done:
            # F2L done + OLL done + not solved => PLL
            return "PLL"
        else:
            # F2L done but U face not uniform => OLL
            return "OLL"

    # Check if D-face cross edges are in place (4 edge stickers on D match D center)
    d_edges = [d_grid[0][1], d_grid[1][0], d_grid[1][2], d_grid[2][1]]
    cross_done = all(e == d_center for e in d_edges)
    if cross_done:
        return "F2L"

    return "CROSS"


def build_prompt_tokens(size: int, cube: Cube, history: list[Move] | None = None) -> list[str]:
    tokens = ["<TASK_POLICY>", "<SIZE>", *int_to_digit_tokens(size), "</SIZE>"]
    # Add stage token: CFOP for 3x3, reduction for 4x4
    if size == 3:
        stage = detect_cfop_stage(cube)
        tokens.append(f"STAGE_{stage}")
    elif size == 4:
        tokens.append(reduction_stage_444_token(cube))
    # Flat sticker colors in fixed face/row/col order (URFDLB)
    for face in FACE_ORDER:
        for row in cube.face_grid(face):
            tokens.extend(f"COL_{color}" for color in row)
    # Last few moves as action history (reduces oscillation)
    if history:
        for move in history[-3:]:
            prefix = "WMOVE" if move.width >= 2 else "MOVE"
            tokens.append(f"{prefix}_{move.face}_{move.turn_name()}")
    tokens.append("<TARGET>")
    return tokens


def build_answer_tokens(move: Move | None) -> list[str]:
    """Single-token answer: MOVE_face_turn, WMOVE_face_turn, or <DONE>."""
    if move is None:
        return ["<DONE>"]
    prefix = "WMOVE" if move.width >= 2 else "MOVE"
    return [f"{prefix}_{move.face}_{move.turn_name()}"]


def parse_answer_tokens(tokens: list[str]) -> Move | None:
    if not tokens:
        raise ValueError("Cannot parse empty answer token list")
    if tokens[0] == "<DONE>":
        return None
    if tokens[0].startswith("WMOVE_"):
        parts = tokens[0].split("_")
        face = parts[1]
        turn_name = parts[2]
        turns = {"CW": 1, "CCW": -1, "HALF": 2}[turn_name]
        return Move(face=face, depth=1, width=2, turns=turns)
    if tokens[0].startswith("MOVE_"):
        parts = tokens[0].split("_")
        face = parts[1]
        turn_name = parts[2]
        turns = {"CW": 1, "CCW": -1, "HALF": 2}[turn_name]
        return Move(face=face, depth=1, width=1, turns=turns)
    raise ValueError(f"Expected MOVE_*/WMOVE_* or <DONE>, got {tokens[0]}")


def build_subgoal_training_examples(
    size: int,
    scramble: tuple[Move, ...],
    staged_solution: list[tuple[str, "Move"]],
) -> list[tuple[list[str], list[str], int]]:
    """Build training examples with sub-goal structure.

    Each CFOP stage is a separate sub-task. distance_to_goal is the
    number of moves until the current sub-goal (stage) is complete,
    NOT until the whole cube is solved. A <DONE> is emitted at the
    end of each stage transition.

    Uses the teacher's stage labels directly (not detect_cfop_stage)
    because Y rotations during CFOP solving reorient the cube.
    """
    cube = Cube(size)
    cube.apply_moves(scramble)
    examples: list[tuple[list[str], list[str], int]] = []
    history: list[Move] = []

    # Group consecutive moves by stage
    segments: list[tuple[str, list[Move]]] = []
    for stage, move in staged_solution:
        if not segments or segments[-1][0] != stage:
            segments.append((stage, []))
        segments[-1][1].append(move)

    for seg_idx, (stage, moves) in enumerate(segments):
        n_moves = len(moves)
        for i, move in enumerate(moves):
            # Build prompt with teacher's stage label (override detect_cfop_stage)
            prompt_tokens = _build_prompt_with_stage(size, cube, stage, history=history)
            answer_tokens = build_answer_tokens(move)
            examples.append((prompt_tokens, answer_tokens, n_moves - i))
            cube.apply_move(move)
            history.append(move)

        # Emit DONE at end of each stage (sub-goal complete)
        next_stage = segments[seg_idx + 1][0] if seg_idx + 1 < len(segments) else "SOLVED"
        prompt_tokens = _build_prompt_with_stage(size, cube, next_stage, history=history)
        examples.append((prompt_tokens, build_answer_tokens(None), 0))

    return examples


def _build_prompt_with_stage(
    size: int, cube: Cube, stage: str, history: list[Move] | None = None
) -> list[str]:
    """Build prompt tokens with an explicit stage label (not auto-detected)."""
    tokens = ["<TASK_POLICY>", "<SIZE>", *int_to_digit_tokens(size), "</SIZE>"]
    tokens.append(f"STAGE_{stage}")
    for face in FACE_ORDER:
        for row in cube.face_grid(face):
            tokens.extend(f"COL_{color}" for color in row)
    if history:
        for move in history[-3:]:
            tokens.append(f"MOVE_{move.face}_{move.turn_name()}")
    tokens.append("<TARGET>")
    return tokens


def build_training_examples(size: int, scramble: tuple[Move, ...]) -> list[tuple[list[str], list[str]]]:
    solution = tuple(move.inverse() for move in reversed(scramble))
    return build_training_examples_from_solution(size, scramble, solution)


def build_training_examples_from_solution(
    size: int,
    scramble: tuple[Move, ...],
    solution: tuple[Move, ...],
) -> list[tuple[list[str], list[str], int]]:
    """Returns list of (prompt_tokens, answer_tokens, distance_to_goal)."""
    cube = Cube(size)
    cube.apply_moves(scramble)
    examples: list[tuple[list[str], list[str], int]] = []
    history: list[Move] = []
    n = len(solution)
    for i, move in enumerate(solution):
        prompt_tokens = build_prompt_tokens(size, cube, history=history)
        answer_tokens = build_answer_tokens(move)
        examples.append((prompt_tokens, answer_tokens, n - i))
        cube.apply_move(move)
        history.append(move)
    examples.append((build_prompt_tokens(size, cube, history=history), build_answer_tokens(None), 0))
    return examples


def random_scramble(
    size: int,
    length: int,
    rng: random.Random,
    max_depth: int = 2,
    max_width: int = 2,
) -> tuple[Move, ...]:
    max_depth = min(max_depth, size // 2)
    max_width = min(max_width, size // 2)
    if max_depth < 1 or max_width < 1:
        raise ValueError(f"Cube size {size} does not support configured scramble ranges")

    scramble: list[Move] = []
    previous_face = None
    opposite_faces = {
        "U": "D",
        "D": "U",
        "L": "R",
        "R": "L",
        "F": "B",
        "B": "F",
    }
    while len(scramble) < length:
        face = rng.choice(FACE_ORDER)
        if face == previous_face:
            continue
        if previous_face is not None and opposite_faces[face] == previous_face and rng.random() < 0.5:
            continue
        depth = rng.randint(1, max_depth)
        width = rng.randint(1, min(max_width, size // 2 - depth + 1))
        turns = rng.choice((1, -1, 2))
        scramble.append(Move(face=face, depth=depth, width=width, turns=turns))
        previous_face = face
    return tuple(scramble)


def scramble_length_for_size(size: int) -> int:
    return 6 + size * 4


# ---------------------------------------------------------------------------
# 4x4 progress metrics — reduction stage detection
# ---------------------------------------------------------------------------

def centers_done_444(cube: Cube) -> bool:
    """Check if all 4 center stickers on each face are uniform (same color).
    Uses within-face uniformity, not canonical colors, because 4x4 centers are movable."""
    if cube.size != 4:
        raise ValueError(f"centers_done_444 only valid for 4x4, got {cube.size}")
    for face in FACE_ORDER:
        grid = cube.face_grid(face)
        # Center is the inner 2x2 block: grid[1][1], grid[1][2], grid[2][1], grid[2][2]
        center_colors = {grid[1][1], grid[1][2], grid[2][1], grid[2][2]}
        if len(center_colors) != 1:
            return False
    return True


def paired_edge_count_444(cube: Cube) -> int:
    """Count how many of the 12 logical edges have their wing pair matched.
    A 4x4 has 24 wing cubies forming 12 edge pairs. Each pair is 'paired' if
    the two wings that share a logical edge position have matching color sets."""
    if cube.size != 4:
        raise ValueError(f"paired_edge_count_444 only valid for 4x4, got {cube.size}")
    limit = cube.limit  # = 3 for 4x4

    # Group wing cubies by their edge key.
    # A wing cubie has exactly 2 coordinates at ±limit and 1 inner coordinate.
    edge_groups: dict[tuple, list[tuple[str, str]]] = {}
    for (position, normal), color in cube.stickers.items():
        # Check if this sticker is on a wing (edge) piece
        at_limit = sum(1 for c in position if abs(c) == limit)
        if at_limit != 2:
            continue
        # This is a wing sticker — group by edge key
        edge_key = tuple(c if abs(c) == limit else 0 for c in position)
        if edge_key not in edge_groups:
            edge_groups[edge_key] = []
        edge_groups[edge_key].append((color, normal))

    # Each logical edge has 2 positions (inner coords differ).
    # Group edge_keys that share the same limit coords but differ in inner coord.
    edge_key_pairs: dict[tuple, list[tuple]] = {}
    for ek in edge_groups:
        # Canonical: sort by inner coord value
        canonical = tuple(abs(c) if abs(c) == limit else 0 for c in ek)
        if canonical not in edge_key_pairs:
            edge_key_pairs[canonical] = []
        edge_key_pairs[canonical].append(ek)

    paired = 0
    for canonical, keys in edge_key_pairs.items():
        if len(keys) != 2:
            continue
        # Get the colors visible on each wing position (stickers facing outward)
        colors_a = set()
        colors_b = set()
        for (color, normal) in edge_groups[keys[0]]:
            colors_a.add(color)
        for (color, normal) in edge_groups[keys[1]]:
            colors_b.add(color)
        if colors_a == colors_b:
            paired += 1

    return paired


def edges_paired_444(cube: Cube) -> bool:
    """Check if all 12 edge pairs are matched on a 4x4."""
    return paired_edge_count_444(cube) == 12


def reduction_stage_444(cube: Cube) -> int:
    """Determine the reduction stage of a 4x4 cube.
    Returns:
        4 = fully solved
        3 = reduced to 3x3 (centers done AND edges paired)
        2 = edges paired only
        1 = centers done only
        0 = neither
    """
    if cube.size != 4:
        raise ValueError(f"reduction_stage_444 only valid for 4x4, got {cube.size}")
    if cube.is_solved():
        return 4
    cd = centers_done_444(cube)
    ep = edges_paired_444(cube)
    if cd and ep:
        return 3
    if ep:
        return 2
    if cd:
        return 1
    return 0


_STAGE_444_TOKENS = {
    0: "STAGE_444_UNREDUCED",
    1: "STAGE_444_CENTERS",
    2: "STAGE_444_EDGES",
    3: "STAGE_444_REDUCED",
    4: "STAGE_444_SOLVED",
}


def reduction_stage_444_token(cube: Cube) -> str:
    """Get the stage token string for a 4x4 cube's current reduction stage."""
    return _STAGE_444_TOKENS[reduction_stage_444(cube)]


def build_reduction_training_examples_444(
    scramble: tuple[Move, ...],
    solution: tuple[Move, ...],
) -> list[tuple[list[str], list[str], int]]:
    """Build stage-local training examples for a 4x4 episode.

    Walks the solution, detects stage boundaries using reduction_stage_444,
    and assigns distance_to_goal as moves remaining until the next stage
    transition (not total moves to solve).

    Returns list of (prompt_tokens, answer_tokens, stage_local_distance).
    """
    cube = Cube(4)
    cube.apply_moves(scramble)
    examples: list[tuple[list[str], list[str], int]] = []
    history: list[Move] = []

    # First pass: find stage boundaries
    sim = cube.copy()
    stages_at_step = []
    for move in solution:
        stages_at_step.append(reduction_stage_444(sim))
        sim.apply_move(move)
    stages_at_step.append(reduction_stage_444(sim))  # final state

    # Find next stage boundary for each step
    n = len(solution)
    next_boundary = [n] * n  # default: end of solution
    current_boundary = n
    for i in range(n - 1, -1, -1):
        if i + 1 < len(stages_at_step) and stages_at_step[i + 1] != stages_at_step[i]:
            current_boundary = i + 1
        next_boundary[i] = current_boundary

    # Second pass: build examples with stage-local distances
    for i, move in enumerate(solution):
        stage_local_dist = next_boundary[i] - i
        prompt_tokens = build_prompt_tokens(4, cube, history=history)
        answer_tokens = build_answer_tokens(move)
        examples.append((prompt_tokens, answer_tokens, stage_local_dist))
        cube.apply_move(move)
        history.append(move)

    # DONE token at end
    examples.append((
        build_prompt_tokens(4, cube, history=history),
        build_answer_tokens(None),
        0,
    ))

    return examples


# ---------------------------------------------------------------------------
# Symmetry augmentation: 24 rotational symmetries of the cube
# ---------------------------------------------------------------------------

def _apply_whole_cube_rotation(cube: "Cube", axis: str, quarter_turns: int) -> "Cube":
    """Rotate the entire cube (all stickers) around an axis. Returns a new Cube."""
    rotated = Cube(cube.size)
    rotated.stickers = {}
    qt = quarter_turns % 4
    for (position, normal), color in cube.stickers.items():
        new_pos = rotate_vec(position, axis, qt)
        new_norm = rotate_vec(normal, axis, qt)
        rotated.stickers[(new_pos, new_norm)] = color
    return rotated


def _enumerate_24_rotations() -> list[list[tuple[str, int]]]:
    """Enumerate all 24 rotational symmetries as sequences of (axis, quarter_turns).
    Deduplicates by checking the resulting face permutation."""
    seen: set[tuple[str, ...]] = set()
    rotations: list[list[tuple[str, int]]] = []

    for x in range(4):
        for y in range(4):
            for z in range(4):
                # Compute face permutation under this rotation
                perm = []
                for face in FACE_ORDER:
                    normal = FACE_NORMALS[face]
                    n = normal
                    if x:
                        n = rotate_vec(n, "x", x)
                    if y:
                        n = rotate_vec(n, "y", y)
                    if z:
                        n = rotate_vec(n, "z", z)
                    # Find which face has this normal
                    for f, fn in FACE_NORMALS.items():
                        if fn == n:
                            perm.append(f)
                            break
                perm_key = tuple(perm)
                if perm_key not in seen:
                    seen.add(perm_key)
                    rots = []
                    if x:
                        rots.append(("x", x))
                    if y:
                        rots.append(("y", y))
                    if z:
                        rots.append(("z", z))
                    rotations.append(rots)
    assert len(rotations) == 24, f"Expected 24 rotations, got {len(rotations)}"
    return rotations


def _compute_move_transform_table() -> list[dict[tuple[str, int], tuple[str, int]]]:
    """For each of the 24 rotations, compute how each move (face, turns) transforms.

    The transform is: map the face via the rotation's face permutation, keep
    CW/CCW/HALF unchanged. This follows from proper rotations preserving
    handedness (the conjugation R ∘ M ∘ R^{-1} preserves the turn direction
    relative to the outward face normal).
    """
    rotations = _enumerate_24_rotations()
    normal_to_face = {v: k for k, v in FACE_NORMALS.items()}
    table: list[dict[tuple[str, int], tuple[str, int]]] = []

    for rot in rotations:
        move_map: dict[tuple[str, int], tuple[str, int]] = {}
        for face in FACE_ORDER:
            # Map the face normal under the rotation to find the new face
            normal = FACE_NORMALS[face]
            for axis, qt in rot:
                normal = rotate_vec(normal, axis, qt)
            new_face = normal_to_face[normal]
            # Turns are preserved (proper rotation preserves handedness)
            for turns in (1, -1, 2):
                move_map[(face, turns)] = (new_face, turns)
        table.append(move_map)
    return table


# Precomputed at import time (fast, ~0.1s)
_ROTATIONS_24: list[list[tuple[str, int]]] | None = None
_MOVE_TRANSFORM_TABLE: list[dict[tuple[str, int], tuple[str, int]]] | None = None


def get_symmetry_rotations() -> list[list[tuple[str, int]]]:
    """Get the 24 rotational symmetries. Cached."""
    global _ROTATIONS_24
    if _ROTATIONS_24 is None:
        _ROTATIONS_24 = _enumerate_24_rotations()
    return _ROTATIONS_24


def get_move_transform_table() -> list[dict[tuple[str, int], tuple[str, int]]]:
    """Get the move transformation table for all 24 rotations. Cached."""
    global _MOVE_TRANSFORM_TABLE
    if _MOVE_TRANSFORM_TABLE is None:
        _MOVE_TRANSFORM_TABLE = _compute_move_transform_table()
    return _MOVE_TRANSFORM_TABLE


def transform_move(move: Move, rotation_idx: int) -> Move:
    """Transform a move under one of the 24 rotational symmetries."""
    table = get_move_transform_table()
    new_face, new_turns = table[rotation_idx][(move.face, move.turns)]
    return Move(face=new_face, depth=move.depth, width=move.width, turns=new_turns)


def transform_episode(episode: Episode, rotation_idx: int) -> Episode:
    """Transform an entire episode (scramble + solution) under a rotational symmetry.

    The transformation maps each move M -> R(M) such that:
      R(M)(solved) = R(M(solved))

    This means: applying R(scramble) to solved gives R(scrambled_state),
    and applying R(solution) then gives R(solved) = solved (for 2x2, uniform faces).
    """
    if rotation_idx == 0:
        # Identity rotation (first in the list)
        return episode
    new_scramble = tuple(transform_move(m, rotation_idx) for m in episode.scramble)
    new_solution = tuple(transform_move(m, rotation_idx) for m in episode.solution)
    return Episode(
        size=episode.size,
        scramble=new_scramble,
        solution=new_solution,
        max_rollout_steps=episode.max_rollout_steps,
    )


def make_episode(
    size: int,
    rng: random.Random,
    scramble_length: int | None = None,
    max_depth: int = 2,
    max_width: int = 2,
) -> Episode:
    scramble_length = scramble_length or scramble_length_for_size(size)
    scramble = random_scramble(
        size=size,
        length=scramble_length,
        rng=rng,
        max_depth=max_depth,
        max_width=max_width,
    )
    solution = tuple(move.inverse() for move in reversed(scramble))
    return Episode(
        size=size,
        scramble=scramble,
        solution=solution,
        max_rollout_steps=max(8, len(solution) * 2),
    )
