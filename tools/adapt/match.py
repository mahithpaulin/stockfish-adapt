"""Play candidate-vs-baseline games and record training samples.

Requires python-chess. Every position sample comes from the candidate engine's
``adapt features`` command, which reports the frozen network's raw evaluation
and the projected accumulator features; the adapter's own correction is not
part of the sample.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import chess
import chess.pgn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.adapt import common, engine as engine_module, model as model_module, openings
else:
    from . import common, engine as engine_module, model as model_module, openings

UCIEngine = engine_module.UCIEngine
EngineError = engine_module.EngineError
FeatureProbe = engine_module.FeatureProbe


class MatchError(RuntimeError):
    pass


@dataclass
class EngineSpec:
    path: str
    label: str
    options: dict[str, Any] = field(default_factory=dict)
    supports_adapt: bool = False

    def options_for(self) -> dict[str, Any]:
        return dict(self.options)


@dataclass
class GameRecord:
    game_id: int
    opening: str
    adapt_color: str
    result: str
    termination: str
    plies: int
    valid: bool
    failure: str | None
    moves: list[str]
    samples: list[dict[str, Any]]
    net_hash: str
    net_name: str
    candidate: str
    baseline: str
    nodes: int
    started_at: str
    duration_seconds: float

    def as_json(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "type": "game",
            "game_id": self.game_id,
            "opening": self.opening,
            "adapt_color": self.adapt_color,
            "result": self.result,
            "termination": self.termination,
            "plies": self.plies,
            "valid": self.valid,
            "failure": self.failure,
            "moves": self.moves,
            "samples": self.samples,
            "net_hash": self.net_hash,
            "net_name": self.net_name,
            "candidate": self.candidate,
            "baseline": self.baseline,
            "nodes": self.nodes,
            "started_at": self.started_at,
            "duration_seconds": round(self.duration_seconds, 3),
        }


def _termination(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient-material"
    if board.is_seventyfive_moves():
        return "seventyfive-moves"
    if board.is_fivefold_repetition():
        return "fivefold-repetition"
    return "claim-draw"


def _append_pgn(path: Path, record: GameRecord, board: chess.Board,
                opening: openings.Opening) -> None:
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Stockfish Adapt match"
    game.headers["Site"] = "local"
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    game.headers["Round"] = str(record.game_id)
    game.headers["White"] = "candidate" if record.adapt_color == "white" else "baseline"
    game.headers["Black"] = "candidate" if record.adapt_color == "black" else "baseline"
    game.headers["Result"] = record.result
    game.headers["TimeControl"] = f"nodes {record.nodes}"
    game.headers["Termination"] = record.termination
    game.headers["Opening"] = opening.name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(str(game))
        handle.write("\n\n")
        handle.flush()
        os.fsync(handle.fileno())


def play_game(
    game_id: int,
    opening: openings.Opening,
    candidate: UCIEngine,
    baseline: UCIEngine,
    *,
    candidate_spec: EngineSpec,
    baseline_spec: EngineSpec,
    nodes: int,
    max_plies: int,
    sample_every: int,
    max_samples_per_game: int,
    move_timeout: float,
    net_hash: str,
    net_name: str,
) -> tuple[GameRecord, chess.Board]:
    started_at = common.utc_now()
    clock = time.monotonic()
    board = chess.Board()
    for move_text in opening.moves:
        move = chess.Move.from_uci(move_text)
        if not board.is_legal(move):
            raise MatchError(
                f"opening {opening.name!r} contains illegal move {move_text} at ply {board.ply()}"
            )
        board.push(move)

    adapt_color = "white" if game_id % 2 == 0 else "black"
    samples: list[dict[str, Any]] = []
    moves: list[str] = []
    termination = ""
    failure: str | None = None

    while board.ply() < max_plies:
        if board.is_game_over(claim_draw=True):
            termination = _termination(board)
            break

        stm_is_candidate = (board.turn == chess.WHITE) == (adapt_color == "white")
        active = candidate if stm_is_candidate else baseline
        active_label = candidate_spec.label if stm_is_candidate else baseline_spec.label

        if (active is candidate and sample_every > 0
                and (board.ply() // 2) % sample_every == 0
                and len(samples) < max_samples_per_game
                and not board.is_check()):
            try:
                active.set_position(board.fen())
                probe = active.probe_features()
                if probe.net_hash.lower() != net_hash.lower():
                    raise MatchError("candidate network hash changed during the match")
                samples.append({
                    "ply": board.ply(),
                    "fen": board.fen(),
                    "raw_eval": probe.raw_eval,
                    "features": probe.features,
                    "stm": "white" if board.turn == chess.WHITE else "black",
                    "in_check": False,
                })
            except EngineError as error:
                failure = f"feature probe failed on {active_label}: {error}"
                termination = "engine-error"
                break

        try:
            active.set_position(board.fen())
            search = active.go(nodes=nodes, timeout=move_timeout)
        except EngineError as error:
            failure = f"{active_label} failed at ply {board.ply()}: {error}"
            termination = "engine-error"
            break

        if not search.bestmove:
            failure = f"{active_label} returned no move at ply {board.ply()}"
            termination = "engine-error"
            break
        try:
            move = chess.Move.from_uci(search.bestmove)
        except ValueError:
            move = None
        if move is None or move not in board.legal_moves:
            failure = f"{active_label} returned illegal move {search.bestmove!r} at ply {board.ply()}"
            termination = "invalid-move"
            break

        board.push(move)
        moves.append(move.uci())

    if not termination:
        termination = _termination(board) if board.is_game_over(claim_draw=True) else "max-plies"

    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
    else:
        result = "*"

    valid = failure is None and termination not in ("max-plies", "engine-error", "invalid-move")
    record = GameRecord(
        game_id=game_id,
        opening=opening.name,
        adapt_color=adapt_color,
        result=result,
        termination=termination,
        plies=board.ply(),
        valid=valid,
        failure=failure,
        moves=moves,
        samples=samples,
        net_hash=net_hash,
        net_name=net_name,
        candidate=candidate_spec.label,
        baseline=baseline_spec.label,
        nodes=nodes,
        started_at=started_at,
        duration_seconds=time.monotonic() - clock,
    )
    return record, board


def _summary(records: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    valid_records = [record for record in records if record.get("valid")]
    result = common.summarize_results(valid_records, "adapt_color")
    failures = [record for record in records if record.get("failure")]
    termination: dict[str, int] = {}
    for record in records:
        key = str(record.get("termination", "unknown"))
        termination[key] = termination.get(key, 0) + 1
    result.update({
        "schema": 1,
        "type": "match-summary",
        "generated_at": common.utc_now(),
        "metadata": metadata,
        "games_requested": metadata.get("games"),
        "games_recorded": len(records),
        "failures": len(failures),
        "failure_examples": [
            {"game_id": record["game_id"], "failure": record["failure"]}
            for record in failures[:10]
        ],
        "termination": dict(sorted(termination.items())),
    })
    return result


def _checkpoint_metadata(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    summary = model_module.checkpoint_summary(path)
    return {
        "path": str(path),
        "sha256": common.sha256_file(path),
        "net_hash": summary["net_hash"],
        "net_name": summary["net_name"],
        "param_count": summary["param_count"],
        "feature_version": summary["feature_version"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.net_hash:
        probe = UCIEngine(args.candidate, {"Threads": 1, "Hash": args.hash_mb},
                          name="net-probe")
        try:
            args.net_hash = probe.net_id()
        finally:
            probe.close()
    if not args.net_hash:
        raise MatchError("could not determine the candidate network hash")
    jsonl_path = output / args.jsonl_name
    pgn_path = output / args.pgn_name
    summary_path = output / args.summary_name
    common.repair_jsonl(jsonl_path)
    existing = common.read_jsonl(jsonl_path)
    existing_ids = {int(record["game_id"]) for record in existing if "game_id" in record}
    if len(existing_ids) != len(existing):
        raise MatchError("duplicate game ids in existing output")
    for record in existing:
        if record.get("net_hash") and record["net_hash"].lower() != args.net_hash.lower():
            raise MatchError("existing output was generated with a different network")
        if int(record.get("nodes", args.nodes)) != args.nodes:
            raise MatchError("existing output uses a different node limit")

    checkpoint = _checkpoint_metadata(args.adapt_file)
    if checkpoint and checkpoint["net_hash"].lower() != args.net_hash.lower():
        raise MatchError(
            f"adapter checkpoint net hash {checkpoint['net_hash']} does not match "
            f"engine network {args.net_hash}"
        )
    if args.candidate_mode in ("ADAPT", "LEARN") and not args.adapt_file:
        raise MatchError(f"candidate mode {args.candidate_mode} requires --adapt-file")
    if checkpoint.get("net_name"):
        args.net_name = checkpoint["net_name"]

    opening_set = openings.get_openings(args.openings)
    candidate_options = {
        "Threads": 1,
        "Hash": args.hash_mb,
    }
    if args.adapt_file:
        candidate_options["AdaptFile"] = str(Path(args.adapt_file).resolve())
        candidate_options["AdaptMode"] = args.candidate_mode
    elif args.candidate_mode != "OFF":
        candidate_options["AdaptMode"] = args.candidate_mode
    baseline_options = {"Threads": 1, "Hash": args.hash_mb}

    candidate_spec = EngineSpec(args.candidate, "candidate", candidate_options, True)
    baseline_spec = EngineSpec(args.baseline, "baseline", baseline_options, False)

    metadata = {
        "candidate": str(Path(args.candidate).resolve()),
        "baseline": str(Path(args.baseline).resolve()),
        "candidate_mode": args.candidate_mode,
        "adapt_file": checkpoint,
        "net_hash": args.net_hash,
        "net_name": args.net_name,
        "nodes": args.nodes,
        "hash_mb": args.hash_mb,
        "games": args.games,
        "openings": args.openings,
        "random_opening_plies": args.random_opening_plies,
        "seed": args.seed,
        "max_plies": args.max_plies,
        "sample_every": args.sample_every,
        "max_samples_per_game": args.max_samples_per_game,
        "started_at": common.utc_now(),
    }

    candidate: UCIEngine | None = None
    baseline: UCIEngine | None = None
    consecutive_failures = 0

    def ensure_engines() -> tuple[UCIEngine, UCIEngine]:
        nonlocal candidate, baseline
        if candidate is None:
            candidate = UCIEngine(
                candidate_spec.path, candidate_spec.options_for(), name="candidate",
                stderr_path=output / "candidate.stderr.log",
            )
        if baseline is None:
            baseline = UCIEngine(
                baseline_spec.path, baseline_spec.options_for(), name="baseline",
                stderr_path=output / "baseline.stderr.log",
            )
        return candidate, baseline

    def drop_engines() -> None:
        nonlocal candidate, baseline
        if candidate is not None:
            candidate.close()
            candidate = None
        if baseline is not None:
            baseline.close()
            baseline = None

    records = list(existing)
    try:
        for game_id in range(args.games):
            if game_id in existing_ids:
                continue
            pair_id = game_id // 2
            opening = opening_set[pair_id % len(opening_set)]
            if args.random_opening_plies:
                opening = openings.randomized_opening(
                    opening,
                    random.Random(args.seed * 1000003 + pair_id),
                    args.random_opening_plies,
                    f"r{pair_id}",
                )
            try:
                candidate, baseline = ensure_engines()
                candidate.new_game()
                baseline.new_game()
                record, board = play_game(
                    game_id,
                    opening,
                    candidate,
                    baseline,
                    candidate_spec=candidate_spec,
                    baseline_spec=baseline_spec,
                    nodes=args.nodes,
                    max_plies=args.max_plies,
                    sample_every=args.sample_every,
                    max_samples_per_game=args.max_samples_per_game,
                    move_timeout=args.move_timeout,
                    net_hash=args.net_hash,
                    net_name=args.net_name,
                )
            except (EngineError, MatchError) as error:
                record = GameRecord(
                    game_id=game_id,
                    opening=opening.name,
                    adapt_color="white" if game_id % 2 == 0 else "black",
                    result="*",
                    termination="engine-error",
                    plies=0,
                    valid=False,
                    failure=str(error),
                    moves=[],
                    samples=[],
                    net_hash=args.net_hash,
                    net_name=args.net_name,
                    candidate=candidate_spec.label,
                    baseline=baseline_spec.label,
                    nodes=args.nodes,
                    started_at=common.utc_now(),
                    duration_seconds=0.0,
                )
                board = chess.Board()
                drop_engines()
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            common.append_jsonl(jsonl_path, record.as_json())
            records.append(record.as_json())
            _append_pgn(pgn_path, record, board, opening)
            print(
                f"game {game_id + 1}/{args.games} {record.adapt_color} "
                f"{record.result} {record.termination} plies={record.plies} "
                f"samples={len(record.samples)} valid={record.valid}"
                + (f" failure={record.failure}" if record.failure else ""),
                flush=True,
            )

            if consecutive_failures >= args.max_consecutive_failures:
                raise MatchError(
                    f"aborting after {consecutive_failures} consecutive engine failures"
                )
    finally:
        drop_engines()

    summary = _summary(records, metadata)
    common.write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Stockfish Adapt games and samples")
    parser.add_argument("--candidate", required=True, help="engine with adapter support")
    parser.add_argument("--baseline", required=True, help="reference engine")
    parser.add_argument("--adapt-file", help="SFA1 checkpoint for the candidate")
    parser.add_argument("--candidate-mode", choices=("OFF", "ADAPT", "LEARN"), default="ADAPT")
    parser.add_argument("--net-hash", help="expected 16-hex network hash (default: query engine)")
    parser.add_argument("--net-name", default="stockfish-default")
    parser.add_argument("--output", required=True)
    parser.add_argument("--jsonl-name", default="games.jsonl")
    parser.add_argument("--pgn-name", default="games.pgn")
    parser.add_argument("--summary-name", default="summary.json")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--nodes", type=int, default=2000)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--openings", choices=("train", "validation"), default="train")
    parser.add_argument("--random-opening-plies", type=int, default=0,
                        help="append seeded legal moves to each opening line")
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--sample-every", type=int, default=4,
                        help="sample every N candidate moves")
    parser.add_argument("--max-samples-per-game", type=int, default=80)
    parser.add_argument("--move-timeout", type=float, default=30.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games <= 0:
        print("--games must be positive", file=sys.stderr)
        return 2
    if args.nodes <= 0:
        print("--nodes must be positive", file=sys.stderr)
        return 2
    if args.sample_every < 0:
        print("--sample-every must be non-negative", file=sys.stderr)
        return 2
    if args.random_opening_plies < 0:
        print("--random-opening-plies must be non-negative", file=sys.stderr)
        return 2
    try:
        summary = run(args)
    except (MatchError, EngineError, model_module.ModelError) as error:
        print(f"match failed: {error}", file=sys.stderr)
        return 2
    print(
        f"recorded {summary['games_recorded']}/{args.games} games; "
        f"valid {summary['games']} failures {summary['failures']} "
        f"score {summary['score_pct']:.3f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
