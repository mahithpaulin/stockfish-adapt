"""Deterministic, balanced opening lines for training and validation.

Training and validation use disjoint lists. Each match game starts from the
standard initial position; the opening line is pushed before play. Within a
batch, consecutive game ids share an opening and swap colours.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class Opening:
    name: str
    moves: tuple[str, ...]


def _opening(name: str, moves: str) -> Opening:
    return Opening(name, tuple(moves.split()))


TRAIN_OPENINGS: tuple[Opening, ...] = (
    _opening("ruy-lopez-closed", "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7"),
    _opening("italian-giuoco", "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d3 d7d6"),
    _opening("scotch-four-knights", "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6 b1c3 f8b4"),
    _opening("najdorf-classical", "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6"),
    _opening("accelerated-dragon", "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g7g6 b1c3 f8g7"),
    _opening("caro-kann-classical", "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5 e4g3 f5g6"),
    _opening("french-steinitz", "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 e4e5 f6d7 f2f4 c7c5"),
    _opening("qgd-classical", "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8"),
    _opening("semi-slav", "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4 a2a4 c8f5"),
    _opening("kings-indian-classical", "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8"),
    _opening("queens-indian", "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 g2g3 c8b7 f1g2 f8e7"),
    _opening("tarrasch-qgd", "d2d4 d7d5 c2c4 e7e6 b1c3 c7c5 c4d5 e6d5 g1f3 b8c6"),
    _opening("reti-main", "g1f3 d7d5 g2g3 g8f6 f1g2 c7c5 e1g1 b8c6 d2d4 e7e6"),
    _opening("english-symmetrical", "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 d7d5 c4d5 f6d5"),
    _opening("petrov-classical", "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4 d2d4 d6d5"),
    _opening("kings-gambit", "e2e4 e7e5 f2f4 e5f4 g1f3 d7d5 e4d5 g8f6 f1c4 f6d5"),
    _opening("scandinavian", "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5 d2d4 g8f6 g1f3 c7c6"),
    _opening("alekhine", "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6 g1f3 d6e5 f3e5 c7c6"),
    _opening("french-classical", "d2d4 e7e6 e2e4 d7d5 b1c3 g8f6 e4e5 f6d7 f2f4 c7c5"),
    _opening("london-system", "d2d4 d7d5 c1f4 g8f6 e2e3 e7e6 g1f3 c7c5 c2c3 b8c6"),
    _opening("symmetrical-english", "c2c4 c7c5 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7 g1f3 g8f6"),
    _opening("double-fianchetto", "g1f3 g8f6 c2c4 c7c5 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7"),
    _opening("vienna", "e2e4 e7e5 b1c3 g8f6 f2f4 d7d5 f4e5 f6e4 g1f3 f8e7"),
    _opening("dutch-classical", "d2d4 f7f5 g2g3 g8f6 f1g2 g7g6 g1f3 f8g7 e1g1 e8g8"),
)


VALIDATION_OPENINGS: tuple[Opening, ...] = (
    _opening("italian-pianissimo", "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8e7 e1g1 e8g8"),
    _opening("scotch-mieses", "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 f8c5 c1e3 d8f6"),
    _opening("closed-sicilian", "e2e4 c7c5 g1f3 b8c6 f1b5 g7g6 e1g1 f8g7 f1e1 g8f6"),
    _opening("sicilian-taimanov", "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 b8c6 b1c3 d8c7"),
    _opening("qga-classical", "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6 e2e3 e7e6 f1c4 c7c5"),
    _opening("qgd-orthodox", "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 g1f3 f8e7 c1g5 e8g8"),
    _opening("nimzo-indian", "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8 f1d3 d7d5"),
    _opening("queens-indian-bb4", "d2d4 g8f6 g1f3 e7e6 c2c4 b7b6 b1c3 c8b7 e2e4 f8b4"),
    _opening("english-symmetric", "c2c4 c7c5 g1f3 g8f6 b1c3 b8c6 d2d4 c5d4 f3d4 e7e6"),
    _opening("reti-qgd", "g1f3 d7d5 d2d4 g8f6 c2c4 e7e6 b1c3 f8e7 c1g5 e8g8"),
    _opening("modern-pirc", "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 c1e3 f8g7 f2f3 e8g8"),
    _opening("french-winawer", "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5 c7c5 a2a3 b4c3"),
)


def randomized_opening(
    opening: Opening,
    rng: random.Random,
    extra_plies: int,
    label: str,
) -> Opening:
    board = chess.Board()
    moves = list(opening.moves)
    for move_text in moves:
        board.push(chess.Move.from_uci(move_text))
    for _ in range(max(0, extra_plies)):
        if board.is_game_over():
            break
        move = rng.choice(tuple(board.legal_moves))
        board.push(move)
        moves.append(move.uci())
    return Opening(f"{opening.name}-{label}", tuple(moves))


def get_openings(split: str) -> tuple[Opening, ...]:
    if split == "train":
        return TRAIN_OPENINGS
    if split == "validation":
        return VALIDATION_OPENINGS
    raise ValueError(f"unknown split {split!r}; expected 'train' or 'validation'")
