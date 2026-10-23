"""Paired validation and promotion gate for a trained adapter.

Runs two arms with the same games and openings:

  candidate = trained adapter vs baseline
  control   = zero adapter vs baseline (or a supplied current best)

Promotion requires a positive paired score difference and a candidate score
whose 95% interval excludes 0.5. The gate never overwrites the candidate; it
only writes a report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.adapt import common, make_zero_adapter, match, model as model_module
else:
    from . import common, make_zero_adapter, match, model as model_module


class ValidationError(RuntimeError):
    pass


def _arm_arguments(args: argparse.Namespace, checkpoint: Path, output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        candidate=args.engine,
        baseline=args.baseline,
        adapt_file=str(checkpoint),
        candidate_mode="ADAPT",
        net_hash=args.net_hash,
        net_name=args.net_name,
        output=str(output),
        jsonl_name="games.jsonl",
        pgn_name="games.pgn",
        summary_name="summary.json",
        games=args.games,
        nodes=args.nodes,
        hash_mb=args.hash_mb,
        openings="validation",
        random_opening_plies=args.random_opening_plies,
        max_plies=args.max_plies,
        sample_every=0,
        max_samples_per_game=0,
        move_timeout=args.move_timeout,
        max_consecutive_failures=args.max_consecutive_failures,
        seed=args.seed,
    )


def _score(record: dict[str, Any]) -> float | None:
    if not record.get("valid"):
        return None
    return common.game_score(record.get("result", "*"), record.get("adapt_color", ""))


def _score_arm(records: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    ids: list[int] = []
    scores: list[float] = []
    for record in records:
        value = _score(record)
        if value is None:
            continue
        ids.append(int(record["game_id"]))
        scores.append(value)
    return ids, scores


def _wilson(scores: list[float]) -> tuple[float, float]:
    if not scores:
        return (0.0, 1.0)
    return common.wilson_interval(sum(scores), len(scores))


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = Path(args.candidate_adapt_file)
    if not candidate_path.exists():
        raise ValidationError(f"candidate checkpoint does not exist: {candidate_path}")
    header = model_module.read_sfa_header(candidate_path)
    if args.net_hash and header["net_hash"].lower() != args.net_hash.lower():
        raise ValidationError("candidate checkpoint network hash does not match --net-hash")

    if args.control_adapt_file:
        control_path = Path(args.control_adapt_file)
        control_header = model_module.read_sfa_header(control_path)
        if control_header["net_hash"].lower() != header["net_hash"].lower():
            raise ValidationError("control checkpoint was trained for a different network")
        control_source = "supplied"
    else:
        control_path = output / "control-zero.sfa"
        make_zero_adapter.make_zero_adapter(candidate_path, control_path)
        control_source = "zero"

    candidate_dir = output / "candidate-arm"
    control_dir = output / "control-arm"
    candidate_summary = match.run(_arm_arguments(args, candidate_path, candidate_dir))
    control_summary = match.run(_arm_arguments(args, control_path, control_dir))

    candidate_records = common.read_jsonl(candidate_dir / "games.jsonl")
    control_records = common.read_jsonl(control_dir / "games.jsonl")
    _, candidate_scores = _score_arm(candidate_records)
    _, control_scores = _score_arm(control_records)

    candidate_by_id = {int(record["game_id"]): record for record in candidate_records}
    control_by_id = {int(record["game_id"]): record for record in control_records}
    shared = sorted(set(candidate_by_id) & set(control_by_id))
    paired_candidate: list[float] = []
    paired_control: list[float] = []
    for game_id in shared:
        left = _score(candidate_by_id[game_id])
        right = _score(control_by_id[game_id])
        if left is None or right is None:
            continue
        paired_candidate.append(left)
        paired_control.append(right)
    differences = [left - right for left, right in zip(paired_candidate, paired_control)]

    candidate_ci = _wilson(paired_candidate)
    mean_candidate = sum(paired_candidate) / len(paired_candidate) if paired_candidate else 0.0
    mean_control = sum(paired_control) / len(paired_control) if paired_control else 0.0
    mean_difference, diff_low, diff_high = common.bootstrap_mean_ci(differences, seed=args.seed)
    failure_rate = candidate_summary["failures"] / max(1, candidate_summary["games_recorded"])

    candidate_elo = common.elo_from_score(mean_candidate)
    control_elo = common.elo_from_score(mean_control)
    paired_games = len(differences)
    score_gate = paired_games > 0 and candidate_ci[0] > 0.5
    difference_gate = paired_games > 0 and diff_low > 0.0
    failure_gate = failure_rate <= args.max_failure_rate
    games_gate = paired_games >= args.min_games
    promotable = score_gate and difference_gate and failure_gate and games_gate

    report = {
        "schema": 1,
        "type": "validation",
        "generated_at": common.utc_now(),
        "engine": str(Path(args.engine).resolve()),
        "baseline": str(Path(args.baseline).resolve()),
        "net_hash": header["net_hash"],
        "net_name": header["net_name"],
        "candidate_checkpoint": {
            "path": str(candidate_path),
            "sha256": common.sha256_file(candidate_path),
        },
        "control_checkpoint": {
            "path": str(control_path),
            "source": control_source,
            "sha256": common.sha256_file(control_path),
        },
        "nodes": args.nodes,
        "openings": "validation",
        "random_opening_plies": args.random_opening_plies,
        "requested_games": args.games,
        "paired_games": paired_games,
        "candidate": {
            "score": mean_candidate,
            "ci95": list(candidate_ci),
            "elo": candidate_elo,
            "summary": candidate_summary,
        },
        "control": {
            "score": mean_control,
            "ci95": list(_wilson(paired_control)),
            "elo": control_elo,
            "summary": control_summary,
        },
        "paired_difference": {
            "mean": mean_difference,
            "ci95": [diff_low, diff_high],
            "elo_difference": (
                candidate_elo - control_elo
                if candidate_elo is not None and control_elo is not None
                else None
            ),
        },
        "failure_rate": failure_rate,
        "gates": {
            "enough_games": games_gate,
            "candidate_score_above_half": score_gate,
            "positive_paired_difference": difference_gate,
            "failure_rate": failure_gate,
        },
        "promotable": promotable,
        "min_games": args.min_games,
        "max_failure_rate": args.max_failure_rate,
    }
    common.write_json(output / "validation.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and gate a Stockfish Adapt checkpoint")
    parser.add_argument("--engine", required=True, help="engine with adapter support")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate-adapt-file", required=True)
    parser.add_argument("--control-adapt-file", help="current-best checkpoint; default is zero")
    parser.add_argument("--net-hash")
    parser.add_argument("--net-name", default="stockfish-default")
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--min-games", type=int, default=100)
    parser.add_argument("--nodes", type=int, default=2000)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--random-opening-plies", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--move-timeout", type=float, default=30.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--max-failure-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games <= 0 or args.min_games <= 0:
        print("--games and --min-games must be positive", file=sys.stderr)
        return 2
    try:
        report = run(args)
    except (ValidationError, match.MatchError, match.EngineError,
            model_module.ModelError, OSError) as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 2
    print(
        f"paired games {report['paired_games']}: candidate "
        f"{report['candidate']['score']:.3f} control {report['control']['score']:.3f} "
        f"difference {report['paired_difference']['mean']:+.3f} "
        f"CI [{report['paired_difference']['ci95'][0]:+.3f}, "
        f"{report['paired_difference']['ci95'][1]:+.3f}] "
        f"promotable={report['promotable']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
