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


def build_prompt_tokens(size: int, cube: Cube, history: list[Move] | None = None) -> list[str]:
    tokens = ["<TASK_POLICY>", "<SIZE>", *int_to_digit_tokens(size), "</SIZE>"]
    # Flat sticker colors in fixed face/row/col order (URFDLB)
    for face in FACE_ORDER:
        for row in cube.face_grid(face):
            tokens.extend(f"COL_{color}" for color in row)
    # Last few moves as action history (reduces oscillation)
    if history:
        for move in history[-3:]:
            tokens.append(f"MOVE_{move.face}_{move.turn_name()}")
    tokens.append("<TARGET>")
    return tokens


def build_answer_tokens(move: Move | None) -> list[str]:
    """Single-token answer: MOVE_face_turn or <DONE>."""
    if move is None:
        return ["<DONE>"]
    return [f"MOVE_{move.face}_{move.turn_name()}"]


def parse_answer_tokens(tokens: list[str]) -> Move | None:
    if not tokens:
        raise ValueError("Cannot parse empty answer token list")
    if tokens[0] == "<DONE>":
        return None
    if tokens[0].startswith("MOVE_"):
        parts = tokens[0].split("_")
        face = parts[1]
        turn_name = parts[2]
        turns = {"CW": 1, "CCW": -1, "HALF": 2}[turn_name]
        return Move(face=face, depth=1, width=1, turns=turns)
    raise ValueError(f"Expected MOVE_* or <DONE>, got {tokens[0]}")


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
