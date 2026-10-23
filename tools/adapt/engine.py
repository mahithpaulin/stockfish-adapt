"""Minimal, strict UCI engine wrapper for matches and probes.

The wrapper never interprets chess rules; callers must validate moves with a
board. It does enforce timeouts and reports crashes instead of hanging.
"""

from __future__ import annotations

import math
import os
import queue
import shlex
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence


class EngineError(RuntimeError):
    pass


class EngineTimeout(EngineError):
    pass


class EngineCrash(EngineError):
    pass


@dataclass
class SearchResult:
    bestmove: str | None
    ponder: str | None = None
    depth: int | None = None
    score_cp: int | None = None
    score_mate: int | None = None
    nodes: int | None = None
    nps: int | None = None
    time_ms: int | None = None
    info_lines: list[str] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)


@dataclass
class FeatureProbe:
    raw_eval: int
    net_hash: str
    feature_version: int
    features: list[float]


def _parse_int_after(tokens: list[str], key: str) -> int | None:
    try:
        index = tokens.index(key)
    except ValueError:
        return None
    if index + 1 >= len(tokens):
        return None
    try:
        return int(tokens[index + 1])
    except ValueError:
        return None


def _parse_info(line: str) -> dict[str, Any]:
    tokens = line.split()
    info: dict[str, Any] = {"line": line}
    if "depth" in tokens:
        info["depth"] = _parse_int_after(tokens, "depth")
    if "nodes" in tokens:
        info["nodes"] = _parse_int_after(tokens, "nodes")
    if "nps" in tokens:
        info["nps"] = _parse_int_after(tokens, "nps")
    if "time" in tokens:
        info["time_ms"] = _parse_int_after(tokens, "time")
    if "score" in tokens:
        index = tokens.index("score")
        if index + 2 < len(tokens):
            kind, value = tokens[index + 1], tokens[index + 2]
            try:
                number = int(value)
            except ValueError:
                number = None
            if kind == "cp":
                info["score_cp"] = number
            elif kind == "mate":
                info["score_mate"] = number
    return info


class UCIEngine:
    def __init__(
        self,
        binary: str | os.PathLike[str] | Sequence[str],
        options: dict[str, Any] | None = None,
        *,
        name: str | None = None,
        timeout: float = 30.0,
        stderr_path: str | os.PathLike[str] | None = None,
    ):
        if isinstance(binary, (list, tuple)):
            command = [str(part) for part in binary]
            self.binary = command[0]
        else:
            raw_binary = str(binary)
            found = shutil.which(raw_binary)
            self.binary = str(Path(found).resolve()) if found else str(Path(raw_binary).resolve())
            command = [self.binary]
        self._command = command
        self.name = name or Path(command[-1]).name
        self.default_timeout = timeout
        self.id_name: str = ""
        self.declared_options: list[str] = []
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(Path(self.binary).resolve().parent),
            start_new_session=True,
        )
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(
            target=self._read_stderr, args=(stderr_path,), daemon=True
        )
        self._stderr_reader.start()
        self._closed = False
        self._initialize(options or {})

    # Process plumbing -----------------------------------------------------
    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        try:
            for line in self._process.stdout:
                self._queue.put(line.rstrip("\r\n"))
        finally:
            self._queue.put(None)

    def _read_stderr(self, stderr_path: str | os.PathLike[str] | None) -> None:
        assert self._process.stderr is not None
        sink = None
        if stderr_path:
            Path(stderr_path).parent.mkdir(parents=True, exist_ok=True)
            sink = open(stderr_path, "a", encoding="utf-8")
        try:
            for line in self._process.stderr:
                if sink:
                    sink.write(line)
                    sink.flush()
        finally:
            if sink:
                sink.close()

    def send(self, command: str) -> None:
        if self._closed or self._process.poll() is not None:
            raise EngineCrash(f"{self.name}: engine not running")
        assert self._process.stdin is not None
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()

    def read_line(self, timeout: float) -> str:
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty as error:
            self._abort()
            raise EngineTimeout(f"{self.name}: timeout waiting for engine output") from error
        if item is None:
            code = self._process.poll()
            self._closed = True
            raise EngineCrash(f"{self.name}: engine exited (code {code})")
        return item

    def read_until(self, predicate: Callable[[str], bool], timeout: float) -> list[str]:
        deadline = time.monotonic() + timeout
        lines: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._abort()
                raise EngineTimeout(f"{self.name}: timeout waiting for expected output")
            line = self.read_line(remaining)
            lines.append(line)
            if predicate(line):
                return lines

    def _initialize(self, options: dict[str, Any]) -> None:
        self.send("uci")
        lines = self.read_until(lambda line: line.strip() == "uciok", self.default_timeout)
        for line in lines:
            if line.startswith("id name "):
                self.id_name = line[len("id name "):].strip()
            elif line.startswith("option "):
                self.declared_options.append(line)
        self.set_options(options)

    def set_options(self, options: dict[str, Any]) -> None:
        for key, value in options.items():
            text = "true" if value is True else "false" if value is False else str(value)
            self.send(f"setoption name {key} value {text}")
        self.send("isready")
        lines = self.read_until(lambda line: line.strip() == "readyok", self.default_timeout)
        errors = [
            line for line in lines
            if line.startswith("info string") and "Adapt:" in line
        ]
        if errors:
            raise EngineError(f"{self.name}: {errors[0]}")

    def new_game(self) -> None:
        self.send("ucinewgame")
        self.send("isready")
        self.read_until(lambda line: line.strip() == "readyok", self.default_timeout)

    def set_position(self, fen: str, moves: list[str] | tuple[str, ...] = ()) -> None:
        command = f"position fen {fen}"
        if moves:
            command += " moves " + " ".join(moves)
        self.send(command)

    def go(self, *, nodes: int | None = None, movetime_ms: int | None = None,
           depth: int | None = None, timeout: float | None = None) -> SearchResult:
        if sum(value is not None for value in (nodes, movetime_ms, depth)) != 1:
            raise ValueError("exactly one of nodes, movetime_ms, depth is required")
        if nodes is not None:
            command = f"go nodes {int(nodes)}"
        elif movetime_ms is not None:
            command = f"go movetime {int(movetime_ms)}"
        else:
            command = f"go depth {int(depth)}"

        self.send(command)
        limit = timeout if timeout is not None else max(self.default_timeout, 60.0)
        lines = self.read_until(lambda line: line.startswith("bestmove"), limit)

        result = SearchResult(bestmove=None, raw_lines=lines)
        for line in lines:
            if line.startswith("info "):
                parsed = _parse_info(line)
                if parsed.get("depth") is not None:
                    result.depth = parsed.get("depth")
                if parsed.get("nodes") is not None:
                    result.nodes = parsed.get("nodes")
                if parsed.get("nps") is not None:
                    result.nps = parsed.get("nps")
                if parsed.get("time_ms") is not None:
                    result.time_ms = parsed.get("time_ms")
                if parsed.get("score_cp") is not None:
                    result.score_cp = parsed.get("score_cp")
                    result.info_lines.append(line)
                if parsed.get("score_mate") is not None:
                    result.score_mate = parsed.get("score_mate")
                    result.info_lines.append(line)

        best = lines[-1].split()
        if len(best) >= 2:
            result.bestmove = None if best[1] in ("(none)", "none") else best[1]
        if "ponder" in best:
            index = best.index("ponder")
            if index + 1 < len(best):
                result.ponder = best[index + 1]
        return result

    def _read_probe_line(self, prefix: str, timeout: float | None = None) -> str:
        lines = self.read_until(
            lambda line: line.startswith(prefix), timeout or self.default_timeout
        )
        for line in lines:
            if line.startswith("info string") and "Adapt:" in line:
                raise EngineError(f"{self.name}: {line}")
        return lines[-1]

    def status(self) -> str:
        self.send("adapt status")
        return self._read_probe_line("adapt_status ")

    def net_id(self) -> str:
        self.send("adapt netid")
        line = self._read_probe_line("adapt_netid ")
        return line.split(maxsplit=1)[1].strip()

    def probe(self) -> dict[str, str]:
        self.send("adapt probe")
        line = self._read_probe_line("adapt_probe ")
        payload = line.split(maxsplit=1)[1] if " " in line else ""
        values: dict[str, str] = {}
        for token in payload.split():
            key, separator, value = token.partition("=")
            if separator:
                values[key] = value
        return values

    def probe_features(self, fen: str | None = None) -> FeatureProbe:
        if fen is not None:
            self.set_position(fen)
        self.send("adapt features")
        line = self._read_probe_line("adapt_features ")
        tokens = line.split()
        if len(tokens) < 5:
            raise EngineError(f"{self.name}: malformed adapt_features output")
        if tokens[1] == "error":
            raise EngineError(f"{self.name}: feature extraction failed: {' '.join(tokens[2:])}")
        raw_eval = int(tokens[1])
        net_hash = tokens[2]
        feature_version = int(tokens[3])
        dimension = int(tokens[4])
        features = [float(token) for token in tokens[5:]]
        if len(features) != dimension:
            raise EngineError(
                f"{self.name}: expected {dimension} features, received {len(features)}"
            )
        if not all(math.isfinite(value) for value in features):
            raise EngineError(f"{self.name}: non-finite feature value")
        return FeatureProbe(raw_eval, net_hash, feature_version, features)

    # Teardown -------------------------------------------------------------
    def _abort(self) -> None:
        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            self._process.kill()
        self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.poll() is None:
                self._process.stdin.write("quit\n")  # type: ignore[union-attr]
                self._process.stdin.flush()  # type: ignore[union-attr]
                self._process.wait(timeout=3.0)
        except Exception:
            pass
        finally:
            if self._process.poll() is None:
                self._abort()
            for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass

    def __enter__(self) -> "UCIEngine":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
