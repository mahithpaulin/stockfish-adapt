"""Measure fixed-node search time and feature-probe latency."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.adapt import common, engine as engine_module
else:
    from . import common, engine as engine_module

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2PP1N2/PP3PPP/RNBQK2R w KQkq - 0 6",
]


def _measure(engine: engine_module.UCIEngine, mode: str, nodes: int,
             repeats: int) -> dict[str, Any]:
    durations: list[float] = []
    nps: list[float] = []
    engine.set_options({"AdaptMode": mode})
    for fen in FENS:
        for _ in range(repeats):
            engine.new_game()
            engine.set_position(fen)
            started = time.perf_counter()
            result = engine.go(nodes=nodes, timeout=120.0)
            elapsed = time.perf_counter() - started
            durations.append(elapsed)
            if result.nodes and elapsed > 0:
                nps.append(result.nodes / elapsed)
    return {
        "mode": mode,
        "runs": len(durations),
        "median_seconds": statistics.median(durations),
        "mean_seconds": statistics.fmean(durations),
        "median_nps": statistics.median(nps) if nps else None,
    }


def _probe_latency(engine: engine_module.UCIEngine, repeats: int) -> dict[str, Any]:
    durations: list[float] = []
    for fen in FENS:
        for _ in range(repeats):
            engine.set_position(fen)
            started = time.perf_counter()
            engine.probe_features()
            durations.append(time.perf_counter() - started)
    return {
        "runs": len(durations),
        "median_ms": statistics.median(durations) * 1000.0,
        "mean_ms": statistics.fmean(durations) * 1000.0,
    }


def run(engine_path: str, adapt_file: str | None, nodes: int, repeats: int) -> dict[str, Any]:
    options: dict[str, Any] = {"Threads": 1, "Hash": 16}
    if adapt_file:
        options.update({"AdaptFile": str(Path(adapt_file).resolve()), "AdaptMode": "OFF"})
    instance = engine_module.UCIEngine(engine_path, options, name="bench", timeout=120.0)
    try:
        net_hash = instance.net_id()
        off = _measure(instance, "OFF", nodes, repeats)
        result: dict[str, Any] = {
            "schema": 1,
            "generated_at": common.utc_now(),
            "engine": str(Path(engine_path).resolve()),
            "net_hash": net_hash,
            "nodes": nodes,
            "positions": len(FENS),
            "repeats": repeats,
            "off": off,
        }
        if adapt_file:
            result["adapt"] = _measure(instance, "ADAPT", nodes, repeats)
            baseline = off["median_seconds"]
            candidate = result["adapt"]["median_seconds"]
            result["overhead"] = {
                "absolute_seconds": candidate - baseline,
                "ratio": candidate / baseline if baseline else None,
            }
            result["feature_probe"] = _probe_latency(instance, repeats)
        return result
    finally:
        instance.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Stockfish Adapt overhead")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--adapt-file")
    parser.add_argument("--nodes", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args.engine, args.adapt_file, args.nodes, args.repeats)
    if args.output:
        common.write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
