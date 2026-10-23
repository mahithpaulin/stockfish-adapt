# Stockfish Adapt — verified integration design

This document is written against the checked-out source, not against assumptions.

## Baseline identity

| Item | Value |
|---|---|
| Upstream commit | `effdc3fd3f6de6e75fe1374a51a7a7eed66f880b` (2026-10-17) |
| Upstream describe | `stockfish-dev-20261017-effdc3fd` |
| Default net | `nn-c9e480e92425.nnue` (`EvalFileDefaultName` in `src/evaluate.h`) |
| License | GNU GPL v3 (`Copying.txt`), notices preserved |

The net is downloaded by `scripts/net.sh` at build time and is **not** committed.

## Actual evaluation path

```
Search::Worker::evaluate()                 src/search.cpp:1931
  -> Eval::evaluate()                      src/evaluate.cpp:48
       -> Network::evaluate()              src/nnue/network.cpp:141
            -> FeatureTransformer::transform()
                 -> AccumulatorStack::evaluate()   (incremental, Finny tables)
                 -> pairwise-clipped transformed features (1024 u8)
            -> NetworkArchitecture::propagate()    (one of 32 layer stacks)
       -> scale_evaluation()               src/evaluate.cpp:60
            optimism, material correlation, rule50 damping, TB clamp
  -> static eval returned to search       (side-to-move perspective)
```

Verified network dimensions (`src/nnue/nnue_architecture.h`):

* `L1 = 1024` accumulator half-dimensions
* `L2 = 32`, `L3 = 32`, `LayerStacks = 32`
* first layer `1024 -> 32` sparse input, pair/square-clipped activations,
  second layer `64 -> 32`, output layer `96 -> 1` plus a skip from `fc_0`
* `OutputScale = 16`; `Network::evaluate` returns `positional / OutputScale`

## Where the adapter attaches

After `scale_evaluation`, inside `Eval::evaluate`:

```
Value raw = evaluate_raw(...);                 // frozen NNUE + existing scaling
Value delta = Adapt::apply(*adapter, accumulatorStack, pos);
return clamp_to_tb(raw + delta);               // OFF: no branch taken
```

`accumulatorStack.latest().accumulation[pos.side_to_move()]` is the current,
incrementally maintained quantized feature-transformer output: 1024 `int16`
values, side-to-move perspective. `Network::evaluate` has just refreshed it for
this position.

### Why the accumulator and not a hidden NNUE layer

Hidden activations are produced in `NetworkArchitecture::propagate()` in a
function-local `Buffer`, are quantized, and are selected from 32 layer stacks.
There is no supported accessor and no stable identity to serialise against.
Tapping them would mean rewriting the NNUE forward pass and its incremental
cache, which violates the frozen-baseline requirement. The accumulator is the
only internal activation that is (a) already computed, (b) incrementally
maintained, (c) exposed by `AccumulatorStack::latest()`, and (d) stable enough
to version a checkpoint against. This is the documented trade-off: the adapter
sees NNUE's quantized first-stage features, not its deeper activations.

## Adapter mathematics

```
x_i  = clip(accumulator[stm][i] / 512, -8, 8)        i = 0..1023
h_j  = (1/sqrt(16)) * sum_{t=0..15} sign(j,t) * x_idx(j,t)     (fixed projection)
a_k  = tanh( sum_j W1[j,k] h_j + b1[k] )             k = 0..19
z    = sum_k W2[k] a_k + b2
delta = limit_cp * tanh(z)                            (-limit_cp, +limit_cp)
V_adapt = V_stockfish + clamp(round(delta), -limit_cp, +limit_cp)
```

* The 1024 -> 96 projection is a fixed, seeded, sparse +/-1 projection. It is
  **not trainable** and therefore is not part of the parameter budget.
* Trainable parameters: `96*20 + 20 + 20 + 1 = 1961`, under the 2048 budget.
  An optional 125 -> 32 configuration is `125*32 + 32 + 32 + 1 = 4065`, under
  4096. Both are selectable by the checkpoint header; the loader rejects any
  architecture mismatch.
* Initialisation for training: `W1` small random, `b1 = 0`, `W2 = 0`, `b2 = 0`.
  `W2 = b2 = 0` makes the correction exactly zero while keeping `a` non-zero so
  gradients reach `W1` after the first update.
* Perspective: accumulator, raw evaluation, and correction are all
  side-to-move relative. Scores use Stockfish internal units (`PawnValue = 208`,
  `src/types.h:199`), not centipawns.
* Numerical safety: tanh bounds the correction; the integer result is clamped to
  `limit_cp` and the final value to the tablebase range. The adapter's own
  arithmetic uses `float`; non-finite weights are rejected at load time.
* OFF mode bypasses the adapter call entirely, so evaluation is bit-identical to
  upstream.

## Frozen weights

The adapter never writes to `Network`, the accumulator caches, or the `.nnue`
file. Loading a checkpoint verifies the network's `get_content_hash()` against
the hash recorded in the checkpoint. A mismatch, a wrong feature version, a
wrong architecture, or an out-of-budget parameter count is rejected with an
`info string` error and the adapter is not installed.

## Incremental evaluation and cost

`AccumulatorStack` already performs incremental updates for search. The adapter
adds 1536 fixed projection additions, `96*20` multiply-adds, 20 tanh for the
hidden layer, and one output tanh per NNUE evaluation. No incremental NNUE state
is touched. Measure overhead with `tools/adapt/bench.py`, which runs `bench`
with the adapter loaded and zero-initialised so the search tree is identical and
only inference cost differs. Only report numbers produced by that run.

## Search and threading

Search code is unchanged. The adapter is immutable while installed; installation
happens from UCI `setoption` after `wait_for_search_finished`. Evaluation reads
an atomic pointer, so an already running search either sees the old immutable
adapter or the new one, never a torn object. Old adapters are retained until
process exit; this is deliberate, bounded by the number of `setoption` calls.

## Mode semantics

| Mode | Behaviour |
|---|---|
| `OFF` | Upstream evaluation exactly; adapter file may be loaded but is not applied. |
| `ADAPT` | Loaded adapter correction is applied inside evaluation. |
| `LEARN` | Same correction, plus the engine is intended to be used by the match runner, which extracts baseline features with `adapt features` between searches. |

Data extraction and training are offline. No Python, file I/O, or process calls
occur in the evaluation path.

## Pilot result (2026-10-23)

The first real-data run used 200 candidate-vs-baseline games at 2,000 nodes,
24 training openings, 2,352 training samples, 301 validation samples, the
1,961-parameter MVP, 20 epochs, and `delta_l2 = 0.01`. Validation used 100
paired games against the pinned baseline, with the zero adapter as the control:

| Run | Correction limit | Paired games | Candidate score | Control score | Paired difference (95% CI) | Promotable |
|---|---:|---:|---:|---:|---|---|
| MVP, `delta_l2=0.01` | 260 cp | 100 | 30.0% | 50.0% | -20.0% (-30.0%, -9.5%) | no |
| MVP, `delta_l2=0.05` | 60 cp | 96 | 38.5% | 50.0% | -11.5% (-21.9%, -1.0%) | no |
| MVP, `delta_l2=0.50` | 10 cp | 100 | 50.0% | 50.0% | 0.0% (0.0%, 0.0%) | no |

The 10 cp run produced a maximum absolute correction of only 0.12 cp, which
rounded to zero in search and therefore reproduced the control exactly. The
larger runs learned corrections that changed play and lost strength. With two
identical engines, the game result is a weak, nearly symmetric label for the
evaluation of an individual position; the MVP overfits it.

A second experiment used Stockfish 18 (`sf_18`) as the fixed opponent for
training and validation, with two seeded random legal plies appended to every
opening line. This produced 986 distinct games out of 1,000, instead of the 46
distinct games in the first SF18 attempt (where deterministic engines repeated
the same opening/colour pair up to 42 times). With 7,767 training samples, 20
epochs, and 200 paired validation games:

| Candidate score | Control score | Paired difference (95% CI) | Promotable |
|---:|---:|---|---|
| 53.8% | 54.0% | -0.25% (-6.5%, +6.0%) | no |

That is a neutral result within the confidence interval, not evidence of an
improvement. The pilot workflow supports a fixed release opponent
(`opponent_tag`, default `sf_18`) for both training and validation; the pinned
current-dev baseline remains the CI identity reference. More games, opponent
and node-budget diversity, and pre-registered regularisation or early stopping
are still needed before drawing a strength conclusion.
