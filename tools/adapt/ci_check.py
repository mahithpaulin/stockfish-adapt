"""CI smoke checks for a built Stockfish Adapt binary.

Verifies the protocol, zero-correction identity, OFF-mode equivalence and, when
a pinned baseline binary is supplied, that OFF mode is bit-identical to the
unmodified engine in fixed-depth searches.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.adapt import common, engine as engine_module, model as model_module
else:
    from . import common, engine as engine_module, model as model_module

UCIEngine = engine_module.UCIEngine
EngineError = engine_module.EngineError

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2PP1N2/PP3PPP/RNBQK2R w KQkq - 0 6",
]


class CheckError(RuntimeError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def search_key(engine: UCIEngine, fen: str, depth: int) -> tuple[Any, Any, Any]:
    engine.new_game()
    engine.set_position(fen)
    result = engine.go(depth=depth, timeout=60.0)
    return (result.bestmove, result.score_cp, result.score_mate)


def zero_checkpoint(path: Path, net_hash: str) -> None:
    model = model_module.AdapterModel(model_module.mvp_architecture(), seed=0)
    model.params = [0.0] * model.param_count
    model.to_sfa(
        path,
        net_hash=net_hash,
        net_name="stockfish-default",
        engine_id="ci-zero",
        metadata={"role": "ci-zero"},
    )


def run(engine_path: str, baseline_path: str | None, depth: int) -> dict[str, Any]:
    report: dict[str, Any] = {"depth": depth, "fens": len(FENS), "checks": []}
    with tempfile.TemporaryDirectory(prefix="adapt-ci-") as directory:
        workspace = Path(directory)
        engine = UCIEngine(engine_path, {"Threads": 1, "Hash": 16}, name="adapt")
        try:
            net_hash = engine.net_id()
            check(len(net_hash) == 16 and all(c in "0123456789abcdef" for c in net_hash.lower()),
                  f"invalid network hash {net_hash!r}")
            report["net_hash"] = net_hash

            probe = engine.probe_features(FENS[0])
            check(probe.feature_version == 1, "unexpected feature version")
            check(probe.net_hash.lower() == net_hash.lower(), "feature network hash mismatch")
            check(len(probe.features) == 96, "default feature dimension must be 96")
            check(all(math.isfinite(value) for value in probe.features), "non-finite feature")
            report["checks"].append("features")

            zero_path = workspace / "zero.sfa"
            zero_checkpoint(zero_path, net_hash)
            engine.set_options({"AdaptFile": str(zero_path), "AdaptMode": "ADAPT"})
            values = engine.probe()
            check(values.get("params") == "1961", "zero adapter was not loaded")
            check(values.get("delta") == "0", f"zero adapter changed eval: {values}")
            check(values.get("corrected") == values.get("baseline"), "zero correction mismatch")
            report["checks"].append("zero-identity")

            off_depth_keys = []
            engine.set_options({"AdaptMode": "OFF"})
            for fen in FENS:
                off_depth_keys.append(search_key(engine, fen, depth))
            report["off_search_keys"] = off_depth_keys

            engine.set_options({"AdaptMode": "ADAPT"})
            adapt_keys = [search_key(engine, fen, depth) for fen in FENS]
            check(adapt_keys == off_depth_keys, "zero adapter changed fixed-depth search")
            report["checks"].append("zero-search-identity")
            engine.set_options({"AdaptMode": "OFF"})

            status = engine.status()
            check("mode=OFF" in status, f"unexpected status: {status}")
            report["checks"].append("status")

            wrong = workspace / "wrong.sfa"
            model_module.AdapterModel(model_module.mvp_architecture(), seed=0).to_sfa(
                wrong,
                net_hash="0000000000000000",
                net_name="wrong",
                engine_id="ci-wrong",
                metadata={},
            )
            try:
                engine.set_options({"AdaptFile": str(wrong)})
            except EngineError:
                pass
            else:
                raise CheckError("mismatched network hash was accepted")
            # Recovery: unload and confirm normal operation.
            engine.set_options({"AdaptMode": "OFF"})
            engine.set_position(FENS[0])
            engine.send("isready")
            engine.read_until(lambda line: line == "readyok", 10.0)
            report["checks"].append("net-hash-rejection")
        finally:
            engine.close()

        if baseline_path:
            baseline = UCIEngine(baseline_path, {"Threads": 1, "Hash": 16}, name="baseline")
            try:
                baseline_keys = [search_key(baseline, fen, depth) for fen in FENS]
            finally:
                baseline.close()
            check(off_depth_keys == baseline_keys,
                  "OFF mode differs from the pinned baseline search")
            report["checks"].append("baseline-identity")
            report["baseline_search_keys"] = baseline_keys
        else:
            report["checks"].append("baseline-skipped")

    report["ok"] = True
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stockfish Adapt CI checks")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args.engine, args.baseline, args.depth)
    except (CheckError, EngineError, model_module.ModelError, OSError) as error:
        print(f"ci check failed: {error}", file=sys.stderr)
        return 2
    if args.report:
        common.write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
