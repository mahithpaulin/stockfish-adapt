"""Summarise match JSONL files without external dependencies."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.adapt import common
else:
    from . import common


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": common.percentile(ordered, 0.9),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    games = [record for record in records if "game_id" in record]
    valid = [record for record in games if record.get("valid")]
    failures = [record for record in games if record.get("failure")]
    terminations: dict[str, int] = {}
    results: dict[str, int] = {}
    for record in games:
        terminations[str(record.get("termination"))] = terminations.get(
            str(record.get("termination")), 0) + 1
        results[str(record.get("result"))] = results.get(str(record.get("result")), 0) + 1

    scores: list[float] = []
    by_color: dict[str, list[float]] = {"white": [], "black": []}
    raw_values: list[float] = []
    sample_counts: list[float] = []
    feature_sums: list[float] | None = None
    feature_squares: list[float] | None = None
    feature_count = 0
    for record in games:
        value = common.game_score(record.get("result", "*"), record.get("adapt_color", ""))
        if value is not None:
            scores.append(value)
            by_color.setdefault(record.get("adapt_color", ""), []).append(value)
        samples = record.get("samples") or []
        sample_counts.append(float(len(samples)))
        for sample in samples:
            raw = float(sample.get("raw_eval", 0))
            raw_values.append(raw)
            features = sample.get("features") or []
            if feature_sums is None:
                feature_sums = [0.0] * len(features)
                feature_squares = [0.0] * len(features)
            if len(features) == len(feature_sums):
                for index, feature in enumerate(features):
                    feature_sums[index] += float(feature)
                    feature_squares[index] += float(feature) * float(feature)
                feature_count += 1

    feature_stats = []
    if feature_sums and feature_squares and feature_count:
        for index, total in enumerate(feature_sums):
            mean = total / feature_count
            variance = max(0.0, feature_squares[index] / feature_count - mean * mean)
            feature_stats.append({
                "index": index,
                "mean": mean,
                "std": math.sqrt(variance),
            })

    summary = common.summarize_results(valid, "adapt_color")
    return {
        "schema": 1,
        "type": "profile",
        "games": len(games),
        "valid_games": len(valid),
        "failures": len(failures),
        "failure_examples": [
            {"game_id": record.get("game_id"), "failure": record.get("failure")}
            for record in failures[:10]
        ],
        "terminations": dict(sorted(terminations.items())),
        "results": dict(sorted(results.items())),
        "score": summary,
        "score_by_color": {
            color: {
                "games": len(values),
                "score": sum(values) / len(values) if values else None,
            }
            for color, values in sorted(by_color.items())
        },
        "plies": _percentiles([float(record.get("plies", 0)) for record in games]),
        "samples_per_game": _percentiles(sample_counts),
        "raw_eval": _percentiles(raw_values),
        "features": {
            "dimensions": len(feature_stats),
            "values": feature_count,
            "dead_dimensions": sum(1 for item in feature_stats if item["std"] < 1e-9),
            "low_variance_dimensions": sum(1 for item in feature_stats if item["std"] < 0.01),
            "per_dimension": feature_stats,
        },
    }


def run(paths: list[str | Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(common.read_jsonl(path))
    return profile(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile match data")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args.data)
    if args.output:
        common.write_json(args.output, report)
    print(json.dumps({key: report[key] for key in (
        "games", "valid_games", "failures", "terminations", "score"
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
