"""Deterministic, resumable adapter training (stdlib only)."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.adapt import common, dataset, model as model_module
else:
    from . import common, dataset, model as model_module

AdapterModel = model_module.AdapterModel
AdamW = model_module.AdamW
Architecture = model_module.Architecture
ModelError = model_module.ModelError
NonFiniteError = model_module.NonFiniteError
DEFAULT_TEMPERATURE = model_module.DEFAULT_TEMPERATURE


def _architecture(name: str, limit_cp: float) -> Architecture:
    if name == "mvp":
        return model_module.mvp_architecture(limit_cp)
    if name == "large":
        return model_module.large_architecture(limit_cp)
    raise ValueError(f"unknown architecture {name!r}")


def _load_state(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_state(path: str | Path, state: dict[str, Any]) -> None:
    common.write_json(Path(path), state)


def _learning_rate(base: float, epoch: int, epochs: int) -> float:
    if epochs <= 1:
        return base
    progress = epoch / (epochs - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base * (0.1 + 0.9 * cosine)


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    paths = [Path(path) for path in args.data]
    prepared = dataset.prepare_dataset(
        paths,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        temperature=args.temperature,
    )
    if not prepared.train:
        raise dataset.DatasetError("training split has no usable samples")

    arch = _architecture(args.arch, args.limit_cp)
    feature_dim = len(prepared.train[0]["features"])
    if feature_dim != arch.input_dim:
        raise dataset.DatasetError(
            f"checkpoint architecture expects {arch.input_dim} features, data has {feature_dim}"
        )

    best_validation = math.inf
    start_epoch = 0
    history: list[dict[str, Any]] = []

    if args.resume:
        state = _load_state(args.resume)
        if int(state.get("schema", 0)) != 1:
            raise model_module.ModelError("unsupported training state schema")
        if state.get("net_hash") != prepared.net_hash:
            raise model_module.ModelError("training state was produced for a different network")
        arch_state = state["arch"]
        if (int(arch_state["input_dim"]), int(arch_state["hidden_dim"])) != (
            arch.input_dim, arch.hidden_dim
        ):
            raise model_module.ModelError("training state architecture does not match --arch")
        model = AdapterModel(Architecture(
            int(arch_state["input_dim"]), int(arch_state["hidden_dim"]),
            float(arch_state["limit_cp"])
        ))
        model.set_params([float(value) for value in state["params"]])
        optimizer = AdamW(model.param_count)
        optimizer.load_state_dict(state["optimizer"])
        best_validation = float(state.get("best_validation", math.inf))
        start_epoch = int(state.get("epoch", 0)) + 1
        history = list(state.get("history", []))
        if model.arch.limit_cp != arch.limit_cp:
            raise model_module.ModelError("resume limit_cp does not match --limit-cp")
    else:
        model = AdapterModel(arch, seed=args.seed)
        if args.init_checkpoint:
            initial = AdapterModel.from_sfa(args.init_checkpoint)
            if initial.arch != arch:
                raise model_module.ModelError("initial checkpoint architecture does not match")
            header, _ = model_module.read_sfa(args.init_checkpoint)
            if header["net_hash"] != prepared.net_hash:
                raise model_module.ModelError(
                    "initial checkpoint was trained for a different network"
                )
            model.set_params(initial.params)
        optimizer = AdamW(
            model.param_count,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    if args.max_samples is not None and len(prepared.train) > args.max_samples:
        rng = random.Random(args.seed + 17)
        prepared.train = rng.sample(prepared.train, args.max_samples)

    train_samples = prepared.train
    validation_samples = prepared.validation
    rng = random.Random(args.seed + 101)
    batch_size = max(1, args.batch_size)
    epochs = max(1, args.epochs)

    for epoch in range(start_epoch, epochs):
        lr = _learning_rate(args.lr, epoch, epochs)
        optimizer.lr = lr
        order = list(range(len(train_samples)))
        rng.shuffle(order)
        total_loss = 0.0
        total_weight = 0.0
        batches = 0

        for start in range(0, len(order), batch_size):
            batch = [train_samples[index] for index in order[start:start + batch_size]]
            gradients = [0.0] * model.param_count
            batch_weight = 0.0
            batch_loss = 0.0
            for sample in batch:
                weight = float(sample["weight"])
                batch_loss += model.loss_and_grad(
                    sample["features"],
                    float(sample["raw_eval"]),
                    float(sample["outcome"]),
                    weight,
                    gradients,
                    alpha=args.alpha,
                    temperature=args.temperature,
                    delta_l2=args.delta_l2,
                )
                batch_weight += weight
            if batch_weight <= 0.0:
                continue
            optimizer.step(model.params, gradients, scale=1.0 / batch_weight)
            total_loss += batch_loss
            total_weight += batch_weight
            batches += 1

            if args.log_every and batches % args.log_every == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} batch {batches} "
                    f"loss {batch_loss / batch_weight:.6f} lr {lr:.2e}",
                    flush=True,
                )

        train_metrics = {
            "loss": total_loss / total_weight if total_weight else 0.0,
            "lr": lr,
            "batches": batches,
        }
        validation_metrics = model.evaluate_dataset(
            validation_samples,
            alpha=args.alpha,
            temperature=args.temperature,
            delta_l2=args.delta_l2,
        )
        validation_loss = validation_metrics["loss"]
        improved = validation_loss < best_validation

        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation_metrics,
            "best": improved,
        }
        history.append(record)
        if improved:
            best_validation = validation_loss

        quantized = model.quantized()
        metadata = {
            "alpha": f"{args.alpha:.9g}",
            "temperature": f"{args.temperature:.9g}",
            "delta_l2": f"{args.delta_l2:.9g}",
            "seed": str(args.seed),
            "epoch": str(epoch + 1),
            "train_loss": f"{train_metrics['loss']:.9g}",
            "validation_loss": f"{validation_loss:.9g}",
            "feature_source": "stockfish-adapt",
        }
        if improved:
            quantized.to_sfa(
                args.out,
                net_hash=prepared.net_hash,
                net_name=prepared.net_name or "stockfish-default",
                engine_id=args.engine_id,
                metadata=metadata,
            )
        quantized.to_sfa(
            args.final_out,
            net_hash=prepared.net_hash,
            net_name=prepared.net_name or "stockfish-default",
            engine_id=args.engine_id,
            metadata=metadata,
        )

        state = {
            "schema": 1,
            "arch": {
                "input_dim": arch.input_dim,
                "hidden_dim": arch.hidden_dim,
                "limit_cp": arch.limit_cp,
            },
            "net_hash": prepared.net_hash,
            "net_name": prepared.net_name,
            "params": model.copy_params(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation": best_validation,
            "history": history,
            "config": vars(args),
        }
        _save_state(args.state, state)

        print(
            f"epoch {epoch + 1}/{epochs} train {train_metrics['loss']:.6f} "
            f"validation {validation_loss:.6f} best {best_validation:.6f}",
            flush=True,
        )

    report = {
        "schema": 1,
        "generated_at": common.utc_now(),
        "duration_seconds": round(time.time() - started, 3),
        "dataset": prepared.summary(),
        "data_fingerprint": dataset.dataset_fingerprint(paths),
        "architecture": {
            "input_dim": arch.input_dim,
            "hidden_dim": arch.hidden_dim,
            "param_count": arch.param_count,
            "limit_cp": arch.limit_cp,
        },
        "config": {
            "epochs": epochs,
            "lr": args.lr,
            "batch_size": batch_size,
            "alpha": args.alpha,
            "temperature": args.temperature,
            "delta_l2": args.delta_l2,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "max_samples": args.max_samples,
        },
        "history": history,
        "best_validation_loss": best_validation,
        "output_checkpoint": str(args.out),
        "final_checkpoint": str(args.final_out),
        "state": str(args.state),
    }
    common.write_json(args.report, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Stockfish Adapt adapter")
    parser.add_argument("--data", nargs="+", required=True,
                        help="match JSONL shard(s)")
    parser.add_argument("--out", default="adapt.sfa", help="best checkpoint path")
    parser.add_argument("--final-out", default="adapt-final.sfa", help="last checkpoint path")
    parser.add_argument("--state", default="train-state.json", help="resumable state path")
    parser.add_argument("--report", default="training-report.json")
    parser.add_argument("--resume", help="resume from this state file")
    parser.add_argument("--init-checkpoint", help="initialise weights from this SFA checkpoint")
    parser.add_argument("--arch", choices=("mvp", "large"), default="mvp")
    parser.add_argument("--limit-cp", type=float, default=model_module.DEFAULT_LIMIT_CP)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--delta-l2", type=float, default=1e-2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--engine-id", default="stockfish-adapt-sfa1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = train(args)
    except (dataset.DatasetError, ModelError, NonFiniteError) as error:
        print(f"training failed: {error}", file=sys.stderr)
        return 2
    print(
        f"trained {report['architecture']['param_count']} parameters on "
        f"{report['dataset']['train_samples']} samples; best validation loss "
        f"{report['best_validation_loss']:.6f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
