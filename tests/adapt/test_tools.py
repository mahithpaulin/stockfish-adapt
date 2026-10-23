"""Unit and integration tests for the Python tooling (stdlib + python-chess)."""

from __future__ import annotations

import json
import math
import os
import shlex
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import chess

from tools.adapt import common, dataset, engine, make_zero_adapter, match, model, openings, train

MOCK = TESTS_DIR / "mock_engine.py"
NET_HASH = "0123456789abcdef"


def write_mock_wrapper(directory: Path) -> Path:
    wrapper = directory / "mock-engine"
    wrapper.write_text(
        "#!/bin/sh\nexec "
        + shlex.quote(sys.executable)
        + " "
        + shlex.quote(str(MOCK))
        + ' "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


def synthetic_game(game_id: int, *, result: str = "1-0", termination: str = "checkmate",
                   net_hash: str = NET_HASH, sample_count: int = 12) -> dict:
    samples = []
    for index in range(sample_count):
        samples.append({
            "ply": index * 2,
            "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
            "raw_eval": (index - sample_count // 2) * 25,
            "features": [((index * 17 + feature) % 101 - 50) / 50.0 for feature in range(96)],
            "stm": "white" if index % 2 == 0 else "black",
            "in_check": False,
        })
    return {
        "schema": 1,
        "type": "game",
        "game_id": game_id,
        "opening": f"test-{game_id % 3}",
        "adapt_color": "white" if game_id % 2 == 0 else "black",
        "result": result,
        "termination": termination,
        "plies": sample_count * 2,
        "valid": True,
        "failure": None,
        "moves": ["a2a3"] * sample_count,
        "samples": samples,
        "net_hash": net_hash,
        "net_name": "test.nnue",
        "candidate": "candidate",
        "baseline": "baseline",
        "nodes": 100,
        "started_at": common.utc_now(),
        "duration_seconds": 0.01,
    }


def write_games(path: Path, games: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(game, sort_keys=True) for game in games) + "\n",
        encoding="utf-8",
    )


class OpeningTests(unittest.TestCase):
    def test_all_openings_are_legal(self) -> None:
        for split in ("train", "validation"):
            for opening in openings.get_openings(split):
                board = chess.Board()
                for ply, text in enumerate(opening.moves):
                    move = chess.Move.from_uci(text)
                    self.assertTrue(board.is_legal(move),
                                    f"{split}/{opening.name} illegal at ply {ply}: {text}")
                    board.push(move)

    def test_splits_are_disjoint(self) -> None:
        train_moves = {opening.moves for opening in openings.get_openings("train")}
        validation_moves = {opening.moves for opening in openings.get_openings("validation")}
        self.assertFalse(train_moves & validation_moves)


class ModelTests(unittest.TestCase):
    def test_mvp_budget_and_zero_correction(self) -> None:
        adapter = model.AdapterModel(model.mvp_architecture(), seed=7)
        self.assertEqual(adapter.param_count, 1961)
        self.assertLessEqual(adapter.param_count, 2048)
        features = [0.25] * 96
        self.assertEqual(adapter.delta(features), 0.0)

    def test_large_budget(self) -> None:
        adapter = model.AdapterModel(model.large_architecture(), seed=7)
        self.assertEqual(adapter.param_count, 4065)
        self.assertLessEqual(adapter.param_count, model.MAX_PARAMS)

    def test_gradient_is_finite_and_moves_weights(self) -> None:
        adapter = model.AdapterModel(model.mvp_architecture(), seed=3)
        features = [((index % 13) - 6) / 6.0 for index in range(96)]
        gradients = [0.0] * adapter.param_count
        loss = adapter.loss_and_grad(features, 80.0, 1.0, 1.0, gradients)
        self.assertTrue(math.isfinite(loss))
        self.assertTrue(all(math.isfinite(value) for value in gradients))
        self.assertGreater(sum(abs(value) for value in gradients), 0.0)
        before = adapter.delta(features)
        optimizer = model.AdamW(adapter.param_count)
        optimizer.step(adapter.params, gradients)
        self.assertNotEqual(adapter.delta(features), before)

    def test_sfa_roundtrip(self) -> None:
        adapter = model.AdapterModel(model.mvp_architecture(limit_cp=180.0), seed=11)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.sfa"
            adapter.to_sfa(
                path,
                net_hash=NET_HASH,
                net_name="test.nnue",
                engine_id="test",
                metadata={"alpha": "0.5", "note": "roundtrip"},
            )
            summary = model.checkpoint_summary(path)
            self.assertEqual(summary["param_count"], 1961)
            self.assertEqual(summary["limit_cp"], 180.0)
            self.assertEqual(summary["metadata"]["meta.alpha"], "0.5")
            loaded = model.AdapterModel.from_sfa(path)
            self.assertEqual(loaded.arch, adapter.arch)
            self.assertEqual(loaded.params, [common.f32(value) for value in adapter.params])
            self.assertEqual(loaded.delta([0.1] * 96), adapter.delta([0.1] * 96))

    def test_sfa_rejects_bad_magic_and_short_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.sfa"
            bad.write_text("NOPE\n", encoding="utf-8")
            with self.assertRaises(model.ModelError):
                model.read_sfa(bad)
            good = model.AdapterModel(model.mvp_architecture())
            path = Path(directory) / "short.sfa"
            good.to_sfa(path, net_hash=NET_HASH, net_name="n", engine_id="e", metadata={})
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace("END\n", ""), encoding="utf-8")
            with self.assertRaises(model.ModelError):
                model.read_sfa(path)

    def test_zero_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "best.sfa"
            model.AdapterModel(model.mvp_architecture()).to_sfa(
                source, net_hash=NET_HASH, net_name="n", engine_id="e", metadata={}
            )
            output = make_zero_adapter.make_zero_adapter(source, Path(directory) / "zero.sfa")
            zero = model.AdapterModel.from_sfa(output)
            self.assertEqual(max(abs(value) for value in zero.params), 0.0)
            self.assertEqual(zero.delta([0.1] * 96), 0.0)


class DatasetTests(unittest.TestCase):
    def test_split_and_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            games = [synthetic_game(index) for index in range(6)]
            games[0]["samples"][0]["raw_eval"] = 2500
            games[1]["samples"][0]["in_check"] = True
            write_games(path, games)
            prepared = dataset.prepare_dataset([path], seed=4, validation_fraction=0.34)
            self.assertEqual(len(prepared.train_games), 4)
            self.assertEqual(len(prepared.validation_games), 2)
            self.assertFalse(set(prepared.train_games) & set(prepared.validation_games))
            self.assertTrue(all(sample["weight"] > 0.0 for sample in prepared.train))
            self.assertEqual(prepared.net_hash, NET_HASH)
            self.assertIn("train_sample_mate_range", prepared.skipped)
            self.assertIn("train_sample_in_check", prepared.skipped)

    def test_split_is_deterministic(self) -> None:
        games = [synthetic_game(index) for index in range(8)]
        first = dataset.split_games(games, seed=9, validation_fraction=0.25)
        second = dataset.split_games(games, seed=9, validation_fraction=0.25)
        self.assertEqual([g["game_id"] for g in first[0]], [g["game_id"] for g in second[0]])
        self.assertEqual([g["game_id"] for g in first[1]], [g["game_id"] for g in second[1]])


class EngineTests(unittest.TestCase):
    def test_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wrapper = write_mock_wrapper(Path(directory))
            with engine.UCIEngine([sys.executable, str(MOCK)], name="mock", timeout=10) as instance:
                self.assertEqual(instance.id_name, "MockFish")
                self.assertEqual(instance.net_id(), NET_HASH)
                probe = instance.probe_features()
                self.assertEqual(probe.feature_version, 1)
                self.assertEqual(len(probe.features), 96)
                self.assertTrue(instance.status().startswith("adapt_status "))
                result = instance.go(depth=1, timeout=10)
                self.assertIsNotNone(result.bestmove)
                self.assertTrue(chess.Move.from_uci(result.bestmove) in chess.Board().legal_moves)


class MatchTests(unittest.TestCase):
    def test_match_resume_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            wrapper = write_mock_wrapper(workspace)
            output = workspace / "match"
            argv = [
                "--candidate", str(wrapper),
                "--baseline", str(wrapper),
                "--candidate-mode", "OFF",
                "--net-hash", NET_HASH,
                "--output", str(output),
                "--games", "2",
                "--nodes", "100",
                "--max-plies", "40",
                "--sample-every", "2",
                "--move-timeout", "10",
            ]
            args = match.build_parser().parse_args(argv)
            summary = match.run(args)
            self.assertEqual(summary["games_recorded"], 2)
            self.assertEqual(summary["failures"], 0)
            records = common.read_jsonl(output / "games.jsonl")
            self.assertEqual(len(records), 2)
            self.assertTrue((output / "games.pgn").exists())
            self.assertTrue(all(record["samples"] for record in records))

            second = match.run(match.build_parser().parse_args(argv))
            self.assertEqual(second["games_recorded"], 2)
            self.assertEqual(len(common.read_jsonl(output / "games.jsonl")), 2)

    def test_mismatched_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            wrapper = write_mock_wrapper(workspace)
            checkpoint = workspace / "wrong.sfa"
            model.AdapterModel(model.mvp_architecture()).to_sfa(
                checkpoint, net_hash="0000000000000000", net_name="n", engine_id="e", metadata={}
            )
            args = match.build_parser().parse_args([
                "--candidate", str(wrapper),
                "--baseline", str(wrapper),
                "--adapt-file", str(checkpoint),
                "--candidate-mode", "ADAPT",
                "--net-hash", NET_HASH,
                "--output", str(workspace / "match"),
                "--games", "1",
                "--nodes", "10",
                "--max-plies", "20",
            ])
            with self.assertRaises(match.MatchError):
                match.run(args)


class TrainingTests(unittest.TestCase):
    def test_training_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            data = workspace / "games.jsonl"
            games = []
            for index in range(6):
                result = ["1-0", "0-1", "1/2-1/2"][index % 3]
                games.append(synthetic_game(index, result=result, sample_count=10))
            write_games(data, games)
            base = [
                "--data", str(data),
                "--out", str(workspace / "best.sfa"),
                "--final-out", str(workspace / "final.sfa"),
                "--state", str(workspace / "state.json"),
                "--report", str(workspace / "report.json"),
                "--epochs", "3",
                "--batch-size", "8",
                "--seed", "3",
            ]
            self.assertEqual(train.main(base), 0)
            best = model.AdapterModel.from_sfa(workspace / "best.sfa")
            self.assertEqual(best.arch.param_count, 1961)
            report = json.loads((workspace / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["dataset"]["train_samples"] > 0, True)
            self.assertTrue(math.isfinite(report["best_validation_loss"]))
            state = json.loads((workspace / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["epoch"], 2)

            resumed = base[:]
            resumed[resumed.index("--epochs") + 1] = "4"
            resumed.extend(["--resume", str(workspace / "state.json")])
            self.assertEqual(train.main(resumed), 0)
            state = json.loads((workspace / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["epoch"], 3)
            self.assertTrue((workspace / "best.sfa").exists())

    def test_training_rejects_empty_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            data = workspace / "empty.jsonl"
            data.write_text("", encoding="utf-8")
            self.assertEqual(train.main([
                "--data", str(data),
                "--out", str(workspace / "best.sfa"),
                "--final-out", str(workspace / "final.sfa"),
                "--state", str(workspace / "state.json"),
                "--report", str(workspace / "report.json"),
                "--epochs", "1",
            ]), 2)


if __name__ == "__main__":
    unittest.main()
