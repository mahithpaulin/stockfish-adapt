#!/usr/bin/env python3
"""Tiny deterministic UCI engine used by the Python tooling tests.

It understands enough of the Stockfish Adapt protocol to exercise the wrapper,
match runner and training data path without building Stockfish.
"""

from __future__ import annotations

import hashlib
import random
import sys

import chess

NET_HASH = "0123456789abcdef"
FEATURE_VERSION = 1
INPUT_DIM = 96

board = chess.Board()
options: dict[str, str] = {}
rng = random.Random(0xC0FFEE)


def emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def reset(fen: str, moves: list[str]) -> None:
    global board
    board = chess.Board(fen)
    for text in moves:
        move = chess.Move.from_uci(text)
        if move not in board.legal_moves:
            raise SystemExit(f"mock engine received illegal move {text}")
        board.push(move)


def features() -> list[float]:
    digest = hashlib.sha256(board.fen().encode()).digest()
    return [((digest[i % len(digest)] / 255.0) * 2.0 - 1.0) for i in range(INPUT_DIM)]


def raw_eval() -> int:
    material = 0
    values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
              chess.ROOK: 500, chess.QUEEN: 900}
    for square, piece in board.piece_map().items():
        value = values.get(piece.piece_type, 0)
        material += value if piece.color == chess.WHITE else -value
    score = material if board.turn == chess.WHITE else -material
    return max(-1900, min(1900, int(score)))


def choose_move() -> str | None:
    legal = list(board.legal_moves)
    if not legal:
        return None
    for move in legal:
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return move.uci()
    captures = [move for move in legal if board.is_capture(move)]
    return rng.choice(captures or legal).uci()


for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue
    tokens = line.split()
    command = tokens[0]

    if command == "uci":
        emit("id name MockFish")
        emit("id author Stockfish Adapt tests")
        emit("option name Threads type spin default 1 min 1 max 8")
        emit("option name Hash type spin default 16 min 1 max 256")
        emit("option name AdaptFile type string default <empty>")
        emit("option name AdaptMode type string default OFF")
        emit("uciok")
    elif command == "isready":
        emit("readyok")
    elif command == "setoption":
        try:
            name_index = tokens.index("name")
            value_index = tokens.index("value")
        except ValueError:
            continue
        name = " ".join(tokens[name_index + 1:value_index])
        value = " ".join(tokens[value_index + 1:])
        options[name] = value
    elif command == "ucinewgame":
        board = chess.Board()
    elif command == "position":
        if len(tokens) >= 2 and tokens[1] == "fen":
            try:
                moves_index = tokens.index("moves")
            except ValueError:
                moves_index = len(tokens)
            fen = " ".join(tokens[2:moves_index])
            moves = tokens[moves_index + 1:] if moves_index < len(tokens) else []
            reset(fen, moves)
        elif len(tokens) >= 2 and tokens[1] == "startpos":
            try:
                moves_index = tokens.index("moves")
            except ValueError:
                moves_index = len(tokens)
            moves = tokens[moves_index + 1:] if moves_index < len(tokens) else []
            reset(chess.STARTING_FEN, moves)
    elif command == "go":
        move = choose_move()
        emit(f"info depth 1 score cp {raw_eval()} nodes 100 time 1")
        emit(f"bestmove {move}" if move else "bestmove (none)")
    elif command == "adapt":
        sub = tokens[1] if len(tokens) > 1 else ""
        if sub == "netid":
            emit(f"adapt_netid {NET_HASH}")
        elif sub == "features":
            if board.is_check():
                emit("adapt_features error in-check")
            else:
                values = " ".join(f"{value:.7f}" for value in features())
                emit(f"adapt_features {raw_eval()} {NET_HASH} {FEATURE_VERSION} "
                     f"{INPUT_DIM} {values}")
        elif sub == "probe":
            emit(f"adapt_probe baseline={raw_eval()} delta=0 corrected={raw_eval()} "
                 f"mode={options.get('AdaptMode', 'OFF')} params=0 limit_cp=0 "
                 f"feature_version={FEATURE_VERSION} net={NET_HASH}")
        elif sub == "status":
            emit("adapt_status loaded=0 mode=OFF net=" + NET_HASH)
    elif command == "quit":
        break
