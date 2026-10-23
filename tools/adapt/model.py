"""Low-rank evaluation adapter model and SFA checkpoint I/O.

Stdlib only. The C++ engine evaluates the same graph in float32 after loading a
checkpoint written here. The training objective is documented in docs/ADAPT.md.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import f32, sigmoid

FORMAT_VERSION = 1
FEATURE_VERSION = 1
DEFAULT_INPUTS = 96
DEFAULT_HIDDEN = 20
LARGE_INPUTS = 125
LARGE_HIDDEN = 32
MAX_PARAMS = 4096
DEFAULT_LIMIT_CP = 260.0
DEFAULT_TEMPERATURE = 312.0


class ModelError(RuntimeError):
    pass


class NonFiniteError(ModelError):
    pass


@dataclass(frozen=True)
class Architecture:
    input_dim: int = DEFAULT_INPUTS
    hidden_dim: int = DEFAULT_HIDDEN
    limit_cp: float = DEFAULT_LIMIT_CP

    @property
    def param_count(self) -> int:
        return self.input_dim * self.hidden_dim + self.hidden_dim + self.hidden_dim + 1

    def validate(self) -> None:
        if not (1 <= self.input_dim <= 192 and 1 <= self.hidden_dim <= 64):
            raise ModelError("invalid adapter architecture dimensions")
        if self.param_count > MAX_PARAMS:
            raise ModelError("adapter exceeds the 4096 parameter budget")
        if not (0.0 < self.limit_cp <= 1000.0):
            raise ModelError("invalid correction limit")


def mvp_architecture(limit_cp: float = DEFAULT_LIMIT_CP) -> Architecture:
    return Architecture(DEFAULT_INPUTS, DEFAULT_HIDDEN, limit_cp)


def large_architecture(limit_cp: float = DEFAULT_LIMIT_CP) -> Architecture:
    return Architecture(LARGE_INPUTS, LARGE_HIDDEN, limit_cp)


class AdapterModel:
    def __init__(self, arch: Architecture, seed: int = 1):
        arch.validate()
        self.arch = arch
        self.seed = seed
        n = arch.param_count
        self.params: list[float] = [0.0] * n
        rng = random.Random(seed)
        for i in range(arch.input_dim * arch.hidden_dim):
            self.params[i] = rng.gauss(0.0, 0.02)
        # b1, W2, b2 remain zero: correction is exactly zero at initialisation.

    # Parameter layout -----------------------------------------------------
    @property
    def param_count(self) -> int:
        return len(self.params)

    @property
    def _w1(self) -> int:
        return 0

    @property
    def _b1(self) -> int:
        return self.arch.input_dim * self.arch.hidden_dim

    @property
    def _w2(self) -> int:
        return self._b1 + self.arch.hidden_dim

    @property
    def _b2(self) -> int:
        return self._w2 + self.arch.hidden_dim

    # Forward --------------------------------------------------------------
    def forward(self, features: list[float]) -> tuple[list[float], float]:
        if len(features) != self.arch.input_dim:
            raise ModelError(f"expected {self.arch.input_dim} features, got {len(features)}")
        p = self.params
        input_dim = self.arch.input_dim
        hidden_dim = self.arch.hidden_dim
        b1 = self._b1
        w2 = self._w2

        hidden = [0.0] * hidden_dim
        for k in range(hidden_dim):
            value = p[b1 + k]
            base = self._w1 + k
            for i in range(input_dim):
                value += p[base + i * hidden_dim] * features[i]
            hidden[k] = math.tanh(value)

        value = p[self._b2]
        for k in range(hidden_dim):
            value += p[w2 + k] * hidden[k]
        delta = self.arch.limit_cp * math.tanh(value)
        return hidden, delta

    def delta(self, features: list[float]) -> float:
        return self.forward(features)[1]

    # Objective ------------------------------------------------------------
    @staticmethod
    def target_probability(raw_eval: float, outcome: float, alpha: float,
                           temperature: float) -> float:
        """Blend the game outcome with the frozen model's probability.

        alpha = 1 would make the target equal to the baseline and carry no
        learning signal. alpha = 0 treats every position as labelled solely by
        the game result. The default alpha = 0.5 keeps the baseline as a prior.
        """
        baseline = sigmoid(raw_eval / temperature)
        return (1.0 - alpha) * outcome + alpha * baseline

    def loss_and_grad(
        self,
        features: list[float],
        raw_eval: float,
        outcome: float,
        weight: float,
        grad: list[float],
        *,
        alpha: float = 0.5,
        temperature: float = DEFAULT_TEMPERATURE,
        delta_l2: float = 1e-2,
    ) -> float:
        """Add weight * d(loss)/d(params) to grad and return weight * loss."""
        if len(grad) != self.param_count:
            raise ModelError("gradient length does not match parameter count")
        hidden, delta = self.forward(features)
        limit = self.arch.limit_cp
        p_target = self.target_probability(raw_eval, outcome, alpha, temperature)

        logit = (raw_eval + delta) / temperature
        probability = sigmoid(logit)
        epsilon = 1e-12
        loss = -(p_target * math.log(probability + epsilon)
                 + (1.0 - p_target) * math.log(1.0 - probability + epsilon))
        loss += delta_l2 * (delta / limit) ** 2
        loss *= weight

        if not math.isfinite(loss):
            raise NonFiniteError("non-finite training loss")

        # d loss / d delta
        d_delta = (probability - p_target) / temperature
        d_delta += 2.0 * delta_l2 * delta / (limit * limit)
        d_delta *= weight

        # delta = limit * tanh(z2)
        d_z2 = d_delta * limit * (1.0 - (delta / limit) ** 2)

        p = self.params
        w2 = self._w2
        b1 = self._b1
        hidden_dim = self.arch.hidden_dim
        input_dim = self.arch.input_dim

        for k in range(hidden_dim):
            grad[w2 + k] += d_z2 * hidden[k]
        grad[self._b2] += d_z2

        for k in range(hidden_dim):
            d_a = d_z2 * p[w2 + k]
            d_z1 = d_a * (1.0 - hidden[k] * hidden[k])
            grad[b1 + k] += d_z1
            base = self._w1 + k
            for i in range(input_dim):
                grad[base + i * hidden_dim] += d_z1 * features[i]

        return loss

    def evaluate_dataset(
        self,
        samples: list[dict[str, Any]],
        *,
        alpha: float = 0.5,
        temperature: float = DEFAULT_TEMPERATURE,
        delta_l2: float = 1e-2,
    ) -> dict[str, float]:
        total_loss = 0.0
        total_weight = 0.0
        max_delta = 0.0
        for sample in samples:
            weight = float(sample["weight"])
            features = sample["features"]
            raw = float(sample["raw_eval"])
            outcome = float(sample["outcome"])
            _, delta = self.forward(features)
            p_target = self.target_probability(raw, outcome, alpha, temperature)
            probability = sigmoid((raw + delta) / temperature)
            eps = 1e-12
            loss = -(p_target * math.log(probability + eps)
                     + (1.0 - p_target) * math.log(1.0 - probability + eps))
            loss += delta_l2 * (delta / self.arch.limit_cp) ** 2
            total_loss += weight * loss
            total_weight += weight
            max_delta = max(max_delta, abs(delta))
        return {
            "loss": total_loss / total_weight if total_weight else 0.0,
            "samples": float(len(samples)),
            "weight": total_weight,
            "max_abs_delta": max_delta,
        }

    # Parameter helpers ----------------------------------------------------
    def copy_params(self) -> list[float]:
        return list(self.params)

    def set_params(self, params: list[float]) -> None:
        if len(params) != self.param_count:
            raise ModelError("parameter vector length mismatch")
        if not all(math.isfinite(value) for value in params):
            raise NonFiniteError("non-finite adapter parameter")
        self.params = [float(value) for value in params]

    def quantized(self) -> "AdapterModel":
        clone = AdapterModel.__new__(AdapterModel)
        clone.arch = self.arch
        clone.seed = self.seed
        clone.params = [f32(value) for value in self.params]
        if not all(math.isfinite(value) for value in clone.params):
            raise NonFiniteError("non-finite adapter parameter after float32 quantisation")
        return clone

    def param_norm(self) -> float:
        return math.sqrt(sum(value * value for value in self.params))

    # Serialisation --------------------------------------------------------
    def to_sfa(
        self,
        path: str | Path,
        *,
        net_hash: str,
        net_name: str,
        engine_id: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        return write_sfa(path, self, net_hash=net_hash, net_name=net_name,
                         engine_id=engine_id, metadata=metadata or {})

    @classmethod
    def from_sfa(cls, path: str | Path) -> "AdapterModel":
        header, params = read_sfa(path)
        arch = Architecture(
            int(header["arch_input"]), int(header["arch_hidden"]), float(header["limit_cp"])
        )
        model = cls.__new__(cls)
        model.arch = arch
        model.seed = 0
        model.params = [f32(value) for value in params]
        return model


class AdamW:
    """Small AdamW implementation with explicit finite-value guards."""

    def __init__(
        self,
        parameter_count: int,
        *,
        lr: float = 5e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m = [0.0] * parameter_count
        self.v = [0.0] * parameter_count

    def step(self, params: list[float], grads: list[float], scale: float = 1.0) -> None:
        if len(params) != len(self.m) or len(grads) != len(self.m):
            raise ModelError("optimizer state length mismatch")
        if not all(math.isfinite(value) for value in grads):
            raise NonFiniteError("non-finite gradient")

        self.step_count += 1
        bias1 = 1.0 - self.beta1 ** self.step_count
        bias2 = 1.0 - self.beta2 ** self.step_count

        for i, gradient in enumerate(grads):
            g = gradient * scale
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g
            m_hat = self.m[i] / bias1
            v_hat = self.v[i] / bias2
            update = m_hat / (math.sqrt(v_hat) + self.epsilon)
            update += self.weight_decay * params[i]
            params[i] -= self.lr * update

        if not all(math.isfinite(value) for value in params):
            raise NonFiniteError("non-finite parameter after optimizer step")

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step_count,
            "m": list(self.m),
            "v": list(self.v),
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if len(state.get("m", [])) != len(self.m) or len(state.get("v", [])) != len(self.m):
            raise ModelError("optimizer state length mismatch")
        self.step_count = int(state.get("step", 0))
        self.m = [float(value) for value in state["m"]]
        self.v = [float(value) for value in state["v"]]
        self.lr = float(state.get("lr", self.lr))
        self.beta1 = float(state.get("beta1", self.beta1))
        self.beta2 = float(state.get("beta2", self.beta2))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.weight_decay = float(state.get("weight_decay", self.weight_decay))


def write_sfa(
    path: str | Path,
    model: AdapterModel,
    *,
    net_hash: str,
    net_name: str,
    engine_id: str,
    metadata: dict[str, str],
) -> str:
    model.arch.validate()
    if len(model.params) != model.arch.param_count:
        raise ModelError("parameter count mismatch")
    if not net_hash or any(character not in "0123456789abcdefABCDEF" for character in net_hash):
        raise ModelError("net_hash must be a hexadecimal string")
    if any("\n" in value or "\r" in value for value in metadata.values()):
        raise ModelError("metadata values must be single-line")

    emitted: dict[str, str] = {
        "format_version": str(FORMAT_VERSION),
        "feature_version": str(FEATURE_VERSION),
        "engine_id": engine_id,
        "net_hash": net_hash.lower(),
        "net_name": net_name,
        "arch_input": str(model.arch.input_dim),
        "arch_hidden": str(model.arch.hidden_dim),
        "param_count": str(model.arch.param_count),
        "limit_cp": f"{model.arch.limit_cp:.9g}",
    }
    for key, value in metadata.items():
        if not key.startswith("meta."):
            key = "meta." + key
        emitted[key] = str(value)

    lines = ["SFA1"]
    for key, value in emitted.items():
        lines.append(f"{key} {value}".rstrip())
    lines.append("weights")
    weights = [f32(value) for value in model.params]
    row: list[str] = []
    for value in weights:
        if not math.isfinite(value):
            raise NonFiniteError("cannot serialise a non-finite weight")
        row.append(f"{value:.9g}")
        if len(row) == 8:
            lines.append(" ".join(row))
            row = []
    if row:
        lines.append(" ".join(row))
    lines.append("END")
    text = "\n".join(lines) + "\n"

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    return str(target)


def read_sfa_header(path: str | Path) -> dict[str, Any]:
    header: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.readline().strip()
        if first != "SFA1":
            raise ModelError(f"not an SFA1 checkpoint: {path}")
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line == "weights":
                break
            key, _, value = line.partition(" ")
            if key in header:
                raise ModelError(f"duplicate header key {key}")
            header[key] = value.strip()
    for key in ("format_version", "feature_version", "net_hash", "arch_input",
                "arch_hidden", "param_count", "limit_cp"):
        if key not in header:
            raise ModelError(f"missing header {key}")
    if int(header["format_version"]) != FORMAT_VERSION:
        raise ModelError("unsupported checkpoint format")
    if int(header["feature_version"]) != FEATURE_VERSION:
        raise ModelError("incompatible feature version")
    arch = Architecture(int(header["arch_input"]), int(header["arch_hidden"]),
                        float(header["limit_cp"]))
    arch.validate()
    if int(header["param_count"]) != arch.param_count:
        raise ModelError("param_count does not match architecture")
    return {
        "format_version": int(header["format_version"]),
        "feature_version": int(header["feature_version"]),
        "engine_id": header.get("engine_id", ""),
        "net_hash": header["net_hash"].lower(),
        "net_name": header.get("net_name", ""),
        "arch_input": arch.input_dim,
        "arch_hidden": arch.hidden_dim,
        "limit_cp": arch.limit_cp,
        "param_count": arch.param_count,
        "metadata": {key: value for key, value in header.items() if key.startswith("meta.")},
    }


def read_sfa(path: str | Path) -> tuple[dict[str, Any], list[float]]:
    header = read_sfa_header(path)
    weights: list[float] = []
    saw_end = False
    with open(path, "r", encoding="utf-8") as handle:
        in_weights = False
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if not in_weights:
                if line == "weights":
                    in_weights = True
                continue
            if line == "END":
                saw_end = True
                break
            for token in line.split():
                value = float(token)
                if not math.isfinite(value):
                    raise ModelError("non-finite weight")
                weights.append(value)
    if not saw_end:
        raise ModelError("checkpoint is missing END")
    if len(weights) != header["param_count"]:
        raise ModelError("weight count does not match param_count")
    return header, weights


def checkpoint_summary(path: str | Path) -> dict[str, Any]:
    header = read_sfa_header(path)
    return {
        "path": str(path),
        "net_hash": header["net_hash"],
        "net_name": header["net_name"],
        "arch_input": header["arch_input"],
        "arch_hidden": header["arch_hidden"],
        "param_count": header["param_count"],
        "limit_cp": header["limit_cp"],
        "feature_version": header["feature_version"],
        "metadata": header["metadata"],
    }
