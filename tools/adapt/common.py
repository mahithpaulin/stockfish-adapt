"""Shared utilities for the Stockfish Adapt tooling.

Stdlib only. The match runner additionally requires python-chess.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    atomic_write_json(path, value)


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default


def append_jsonl(path: str | os.PathLike[str], value: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    line = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def repair_jsonl(path: str | os.PathLike[str]) -> None:
    """Drop a trailing partial line left by a crash before appending again."""
    target = Path(path)
    if not target.exists() or target.stat().st_size == 0:
        return
    with open(target, "rb+") as handle:
        data = handle.read()
        if not data.endswith(b"\n"):
            cut = data.rfind(b"\n")
            handle.seek(cut + 1 if cut >= 0 else 0)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(target, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def lround(value: float) -> int:
    """C++ std::lround semantics: round half away from zero."""
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def f32(value: float) -> float:
    """Round a Python float to IEEE-754 binary32, matching C++ float storage."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def elo_from_score(score: float) -> float | None:
    if not 0.0 < score < 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def wilson_interval(wins: float, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return (0.0, 1.0)
    p = wins / games
    denominator = 1.0 + z * z / games
    centre = p + z * z / (2.0 * games)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * games)) / games)
    return (max(0.0, (centre - margin) / denominator),
            min(1.0, (centre + margin) / denominator))


def bootstrap_mean_ci(
    values: Sequence[float], resamples: int = 5000, seed: int = 1, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Percentile bootstrap for the mean. Returns (mean, low, high)."""
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    low_index = int((alpha / 2.0) * (resamples - 1))
    high_index = int((1.0 - alpha / 2.0) * (resamples - 1))
    return (sum(values) / n, means[low_index], means[high_index])


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def game_score(result: str, color: str) -> float | None:
    if result == "1-0":
        return 1.0 if color == "white" else 0.0
    if result == "0-1":
        return 0.0 if color == "white" else 1.0
    if result == "1/2-1/2":
        return 0.5
    return None


def summarize_results(records: Iterable[dict[str, Any]], adapt_side_key: str = "adapt_color") -> dict[str, Any]:
    games = 0
    score = 0.0
    wins = draws = losses = invalid = 0
    for record in records:
        if not record.get("valid", False):
            invalid += 1
            continue
        color = record.get(adapt_side_key)
        value = game_score(record.get("result", "*"), color)
        if value is None:
            invalid += 1
            continue
        games += 1
        score += value
        if value == 1.0:
            wins += 1
        elif value == 0.5:
            draws += 1
        else:
            losses += 1
    pct = score / games if games else 0.0
    low, high = wilson_interval(score, games)
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "invalid": invalid,
        "score": score,
        "score_pct": pct,
        "elo": elo_from_score(pct) if games else None,
        "score_ci95": [low, high],
        "elo_ci95": [elo_from_score(low), elo_from_score(high)],
    }
