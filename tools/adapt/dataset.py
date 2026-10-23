"""Dataset loading, filtering, weighting and deterministic game-level splits."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .common import read_jsonl
from .model import DEFAULT_TEMPERATURE, sigmoid

MAX_ABS_EVAL = 2000
MIN_WEIGHT = 0.05


class DatasetError(RuntimeError):
    pass


@dataclass
class Dataset:
    train: list[dict[str, Any]] = field(default_factory=list)
    validation: list[dict[str, Any]] = field(default_factory=list)
    train_games: list[int] = field(default_factory=list)
    validation_games: list[int] = field(default_factory=list)
    net_hash: str = ""
    net_name: str = ""
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.train) + len(self.validation)

    def summary(self) -> dict[str, Any]:
        return {
            "train_samples": len(self.train),
            "validation_samples": len(self.validation),
            "train_games": len(self.train_games),
            "validation_games": len(self.validation_games),
            "net_hash": self.net_hash,
            "net_name": self.net_name,
            "skipped": dict(sorted(self.skipped.items())),
        }


def _bump(skipped: dict[str, int], key: str, amount: int = 1) -> None:
    skipped[key] = skipped.get(key, 0) + amount


def load_games(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for path in paths:
        for record in read_jsonl(path):
            if record.get("type") == "summary":
                continue
            if "game_id" not in record or "samples" not in record:
                continue
            games.append(record)
    if not games:
        raise DatasetError("no games found in the supplied JSONL files")
    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    for game in games:
        game_id = int(game["game_id"])
        if game_id in seen:
            raise DatasetError(f"duplicate game id {game_id}")
        seen.add(game_id)
        unique.append(game)
    unique.sort(key=lambda game: int(game["game_id"]))
    return unique


def split_games(
    games: list[dict[str, Any]], *, seed: int, validation_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < validation_fraction < 0.5:
        raise DatasetError("validation_fraction must be between 0 and 0.5")
    eligible = [
        game for game in games
        if not game.get("failure") and game.get("termination") not in ("unfinished", "max-plies")
    ]
    if len(eligible) < 2:
        raise DatasetError("need at least two usable games to split train/validation")
    ordered = sorted(eligible, key=lambda game: int(game["game_id"]))
    random.Random(seed).shuffle(ordered)
    validation_count = max(1, min(len(ordered) - 1, round(len(ordered) * validation_fraction)))
    validation_ids = {int(game["game_id"]) for game in ordered[:validation_count]}
    train = [game for game in eligible if int(game["game_id"]) not in validation_ids]
    validation = [game for game in eligible if int(game["game_id"]) in validation_ids]
    return train, validation


def _outcome_stm(game: dict[str, Any], stm: str) -> float | None:
    result = game.get("result")
    if result == "1/2-1/2":
        return 0.5
    if result not in ("1-0", "0-1"):
        return None
    white_won = result == "1-0"
    return 1.0 if (white_won and stm == "white") or (not white_won and stm == "black") else 0.0


def build_samples(
    games: list[dict[str, Any]],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_abs_eval: int = MAX_ABS_EVAL,
    min_weight: float = MIN_WEIGHT,
) -> tuple[list[dict[str, Any]], dict[str, int], str, str]:
    skipped: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    net_hashes: set[str] = set()
    net_name = ""
    feature_dim: int | None = None

    for game in games:
        if game.get("failure"):
            _bump(skipped, "game_failure")
            continue
        if game.get("termination") in ("unfinished", "max-plies"):
            _bump(skipped, "game_unfinished")
            continue
        if game.get("net_hash"):
            net_hashes.add(str(game["net_hash"]).lower())
            net_name = str(game.get("net_name", net_name))

        raw_samples = game.get("samples") or []
        if not raw_samples:
            _bump(skipped, "game_without_samples")
            continue
        prepared: list[dict[str, Any]] = []
        for sample in raw_samples:
            if sample.get("in_check"):
                _bump(skipped, "sample_in_check")
                continue
            stm = sample.get("stm")
            if stm not in ("white", "black"):
                _bump(skipped, "sample_bad_stm")
                continue
            outcome = _outcome_stm(game, stm)
            if outcome is None:
                _bump(skipped, "sample_unknown_result")
                continue
            features = sample.get("features")
            raw_eval = sample.get("raw_eval")
            if not isinstance(features, list) or not features or raw_eval is None:
                _bump(skipped, "sample_malformed")
                continue
            if feature_dim is None:
                feature_dim = len(features)
            elif len(features) != feature_dim:
                raise DatasetError("inconsistent feature dimensions across games")
            try:
                raw = float(raw_eval)
                values = [float(value) for value in features]
            except (TypeError, ValueError):
                _bump(skipped, "sample_malformed")
                continue
            if not math.isfinite(raw) or not all(math.isfinite(value) for value in values):
                _bump(skipped, "sample_non_finite")
                continue
            if abs(raw) >= max_abs_eval:
                _bump(skipped, "sample_mate_range")
                continue
            prepared.append({
                "features": values,
                "raw_eval": raw,
                "outcome": outcome,
                "ply": int(sample.get("ply", 0)),
                "fen": sample.get("fen", ""),
            })
        if not prepared:
            _bump(skipped, "game_without_usable_samples")
            continue

        # Each game contributes equal total weight; uncertain positions within a
        # game carry more signal than positions already decided.
        game_scale = 1.0 / len(prepared)
        for sample in prepared:
            probability = sigmoid(sample["raw_eval"] / temperature)
            uncertainty = max(min_weight, 4.0 * probability * (1.0 - probability))
            sample["weight"] = game_scale * uncertainty
            sample["game_id"] = int(game["game_id"])
            samples.append(sample)

    if net_hashes and len(net_hashes) != 1:
        raise DatasetError(f"mixed networks in training data: {sorted(net_hashes)}")
    if not net_hashes:
        raise DatasetError("training data does not record a network hash")
    if not samples:
        raise DatasetError("no usable samples after filtering")
    return samples, skipped, next(iter(net_hashes)), net_name


def prepare_dataset(
    paths: Iterable[str | Path],
    *,
    seed: int = 1,
    validation_fraction: float = 0.1,
    temperature: float = DEFAULT_TEMPERATURE,
    max_abs_eval: int = MAX_ABS_EVAL,
    min_weight: float = MIN_WEIGHT,
) -> Dataset:
    games = load_games(paths)
    train_games, validation_games = split_games(
        games, seed=seed, validation_fraction=validation_fraction
    )
    train_samples, train_skipped, net_hash, net_name = build_samples(
        train_games, temperature=temperature, max_abs_eval=max_abs_eval, min_weight=min_weight
    )
    validation_samples, validation_skipped, validation_hash, validation_name = build_samples(
        validation_games, temperature=temperature, max_abs_eval=max_abs_eval, min_weight=min_weight
    )
    if validation_hash != net_hash:
        raise DatasetError("train and validation games use different networks")
    if not validation_name:
        validation_name = net_name
    skipped: dict[str, int] = {}
    for prefix, counts in (("train", train_skipped), ("validation", validation_skipped)):
        for key, value in counts.items():
            skipped[f"{prefix}_{key}"] = value
    if not validation_samples:
        raise DatasetError("validation split has no usable samples")
    return Dataset(
        train=train_samples,
        validation=validation_samples,
        train_games=sorted(int(game["game_id"]) for game in train_games),
        validation_games=sorted(int(game["game_id"]) for game in validation_games),
        net_hash=net_hash,
        net_name=validation_name,
        skipped=skipped,
    )


def dataset_fingerprint(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(str(path) for path in paths):
        digest.update(Path(path).name.encode())
        digest.update(b"\0")
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()
