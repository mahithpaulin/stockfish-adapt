"""Create an all-zero SFA1 adapter for control arms."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.adapt import model as model_module
else:
    from . import model as model_module


def make_zero_adapter(source: str | Path, output: str | Path) -> Path:
    header = model_module.read_sfa_header(source)
    arch = model_module.Architecture(
        int(header["arch_input"]), int(header["arch_hidden"]), float(header["limit_cp"])
    )
    model = model_module.AdapterModel(arch, seed=0)
    model.params = [0.0] * arch.param_count
    model.to_sfa(
        output,
        net_hash=header["net_hash"],
        net_name=header.get("net_name", "stockfish-default"),
        engine_id="stockfish-adapt-control",
        metadata={
            "role": "zero-control",
            "source_checkpoint": Path(source).name,
        },
    )
    return Path(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a zero-correction adapter")
    parser.add_argument("--source", required=True, help="checkpoint whose architecture to copy")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = make_zero_adapter(args.source, args.output)
    except (OSError, model_module.ModelError) as error:
        print(f"cannot create zero adapter: {error}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
