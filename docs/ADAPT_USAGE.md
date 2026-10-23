# Stockfish Adapt — usage

Stockfish Adapt keeps the frozen Stockfish NNUE and adds a very small adapter
that corrects its evaluation. The adapter is trained offline from self-play
data; the engine only loads a checkpoint and applies a fixed graph.

Design and verified source details: [ADAPT.md](ADAPT.md).

## Build

```sh
make -C src -j2 build ARCH=x86-64-avx2 COMP=g++
```

`make build` downloads the default network (`nn-c9e480e92425.nnue`). The network
is not part of this repository. The adapter is compiled into the binary; no
Python is needed to run the engine.

Run the C++ adapter unit test without a network:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -I src \
  src/adapt/adapter.cpp tests/adapt/adapter_test.cpp -o adapter_test
./adapter_test
```

## UCI interface

Options:

| Option | Default | Meaning |
|---|---|---|
| `AdaptFile` | empty | Path to an `SFA1` checkpoint. Empty unloads. |
| `AdaptMode` | `OFF` | `OFF`, `ADAPT`, or `LEARN` (case-insensitive). |

Commands:

| Command | Output |
|---|---|
| `adapt features` | `<baseline-eval> <net-hash> <feature-version> <dim> <features...>` |
| `adapt probe` | `baseline=... delta=... corrected=... mode=... params=... ...` |
| `adapt netid` | 16-hex content hash of the loaded network |
| `adapt status` | loaded/path/mode/network summary |

`adapt features` reports the frozen network's scaled evaluation and the
projected 96-d accumulator features for the side to move. It refuses positions
in check, matching the engine's evaluation contract. Features are only valid
for the network hash printed beside them.

Example:

```
uci
setoption name AdaptFile value /path/to/adapt.sfa
setoption name AdaptMode value ADAPT
isready
position startpos
go depth 12
quit
```

`OFF` bypasses the correction and is intended to be bit-identical to the pinned
upstream commit.

## Checkpoints

`SFA1` is a line-oriented text format:

```
SFA1
format_version 1
feature_version 1
engine_id stockfish-adapt-sfa1
net_hash 0123456789abcdef
net_name nn-c9e480e92425.nnue
arch_input 96
arch_hidden 20
param_count 1961
limit_cp 260
meta.alpha 0.5
weights
<1961 floats>
END
```

The loader rejects a wrong network hash, feature version, architecture,
parameter count, or non-finite weight. The 96→20 MVP has 1961 parameters; the
optional 125→32 configuration has 4065. Both are under their budgets. Extra
`meta.*` lines are preserved as header strings.

## Data generation

Requires Python 3.10+ and `python-chess`:

```sh
python -m pip install python-chess
```

Play a match and record samples (`--sample-every` counts candidate moves):

```sh
python tools/adapt/match.py \
  --candidate src/stockfish \
  --baseline /path/to/pinned/stockfish \
  --adapt-file profiles/current.sfa \
  --candidate-mode ADAPT \
  --output runs/match-001 \
  --games 100 \
  --nodes 2000 \
  --openings train \
  --sample-every 4
```

If `--adapt-file` is omitted and `--candidate-mode OFF` is used, the run records
features without loading an adapter. Without `--net-hash`, the runner asks the
candidate engine for `adapt netid`. The output directory contains `games.jsonl`
(resumable, one record per game), `games.pgn`, `summary.json`, and stderr logs.
Re-running the same command resumes missing game ids.

The training signal is the game result from the side-to-move perspective. Each
record stores the frozen network's eval and features before the move. Positions
in check are not sampled. Training and validation use disjoint opening sets;
never generate training data on `--openings validation`.

## Training

```sh
python tools/adapt/train.py \
  --data runs/match-001/games.jsonl \
  --out profiles/candidate.sfa \
  --final-out profiles/candidate-final.sfa \
  --state runs/train-001/state.json \
  --report runs/train-001/report.json \
  --epochs 40
```

The objective is a calibration loss against the game outcome blended with the
baseline prior (`--alpha`), with temperature in internal units (default 312 ≈
1.5 pawns), an L2 penalty on the correction, and per-game weighting. Every game
contributes equal total weight; positions closer to equality contribute more
within a game.

Training is deterministic given `--seed`, saves the best validation epoch to
`--out`, saves the last epoch to `--final-out`, and writes optimizer state for
`--resume`. Game-level train/validation splits are deterministic. Training is
stdlib-only, but it expects the `features` produced by the match runner.

`--init-checkpoint` starts from an existing SFA1 file (still requiring the same
network). `--arch large` selects 125→32.

## Validation and promotion

```sh
python tools/adapt/validate.py \
  --engine src/stockfish \
  --baseline /path/to/pinned/stockfish \
  --candidate-adapt-file profiles/candidate.sfa \
  --output runs/validation-001 \
  --games 1000 \
  --min-games 1000 \
  --nodes 2000
```

This runs a candidate arm and a zero-adapter control arm on the disjoint
validation openings. It reports paired score differences with a bootstrap
interval and a candidate score interval. `promotable` is only true when:

* at least `--min-games` usable paired games exist,
* the candidate score 95% interval is entirely above 0.5,
* the paired candidate-minus-control 95% interval is entirely above 0,
* the candidate failure rate is at most `--max-failure-rate`.

The report is advisory; it never overwrites the candidate checkpoint. A 1,000
game match cannot resolve a very small Elo gain, so a non-promotable result is
not proof that the adapter is worse.

`tools/adapt/make_zero_adapter.py --source candidate.sfa --output zero.sfa`
creates the exact-zero control checkpoint.

## Diagnostics

```sh
python tools/adapt/profile.py --data runs/match-001/games.jsonl
python tools/adapt/bench.py --engine src/stockfish \
  --adapt-file profiles/candidate.sfa --nodes 5000
python tools/adapt/ci_check.py --engine src/stockfish \
  --baseline /path/to/pinned/stockfish
```

`ci_check.py` verifies the protocol, zero-correction identity, network-hash
rejection, and that `OFF` mode matches the pinned baseline in fixed-depth
searches. CI runs the Python tests, the C++ adapter test, a candidate build, and
this check (see `.github/workflows/adapt.yml`).

## Limits

* The adapter sees the quantized 1024-d accumulator, not deeper NNUE layers.
* The projection is fixed; only the small MLP is trained.
* The correction is bounded by `limit_cp` and excludes the tablebase range.
* Search code and NNUE incremental state are untouched.
* Evaluation-time cost is small but nonzero; `bench.py` measures it rather than
  assuming it.
