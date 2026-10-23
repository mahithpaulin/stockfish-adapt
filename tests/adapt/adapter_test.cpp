/*
  Stockfish Adapt adapter unit tests. Standalone: only adapter.cpp is needed.
  Copyright (C) 2026 the Stockfish Adapt contributors. GPLv3 or later.
*/

#include "adapt/adapter.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                                           \
    do                                                                                             \
    {                                                                                              \
        if (!(condition))                                                                          \
        {                                                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);               \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

using Stockfish::Adapt::Adapter;
using Stockfish::Adapt::Spec;

constexpr const char* NetHash = "0123456789abcdef";

std::vector<float> zeroWeights(const Spec& spec) {
    return std::vector<float>(static_cast<std::size_t>(spec.paramCount()), 0.0f);
}

std::array<std::int16_t, Stockfish::Adapt::kAccumulatorDimensions> deterministicAccumulator() {
    std::array<std::int16_t, Stockfish::Adapt::kAccumulatorDimensions> acc{};
    std::uint32_t state = 0x12345678u;
    for (std::size_t i = 0; i < acc.size(); ++i)
    {
        state = state * 1664525u + 1013904223u;
        acc[i] = static_cast<std::int16_t>((state >> 16) % 4001) - 2000;
    }
    return acc;
}

std::string tempPath() {
    const auto  ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    const char* tmp   = std::getenv("TMPDIR");
    const std::string dir = tmp && *tmp ? tmp : "/tmp";
    return dir + "/stockfish_adapt_test_" + std::to_string(ticks) + ".sfa";
}

void testParameterBudget() {
    Spec mvp;
    CHECK(mvp.paramCount() == 96 * 20 + 20 + 20 + 1);
    CHECK(mvp.paramCount() == 1961);
    CHECK(mvp.paramCount() <= 2048);
    CHECK(mvp.valid());

    Spec large;
    large.inputDim  = Stockfish::Adapt::kLargeInputs;
    large.hiddenDim = Stockfish::Adapt::kLargeHidden;
    CHECK(large.paramCount() == 125 * 32 + 32 + 32 + 1);
    CHECK(large.paramCount() == 4065);
    CHECK(large.paramCount() <= 4096);
    CHECK(large.valid());

    Spec tooLarge;
    tooLarge.inputDim  = Stockfish::Adapt::kMaxInputs;
    tooLarge.hiddenDim = Stockfish::Adapt::kMaxHidden;
    CHECK(tooLarge.paramCount() > 4096);
    CHECK(!tooLarge.valid());
}

void testZeroInitialisation() {
    Spec               spec;
    std::vector<float> weights = zeroWeights(spec);
    Adapter            adapter(spec, weights, NetHash, "test.nnue", "test", {});

    auto acc = deterministicAccumulator();
    CHECK(adapter.correction(acc.data()) == 0.0f);

    // Non-zero W1 with zero W2/b2 still must give exactly zero correction.
    for (int i = 0; i < spec.inputDim * spec.hiddenDim; ++i)
        weights[static_cast<std::size_t>(i)] = 0.001f * static_cast<float>(i % 17 - 8);
    Adapter withHidden(spec, weights, NetHash, "test.nnue", "test", {});
    CHECK(withHidden.correction(acc.data()) == 0.0f);
}

void testBoundsAndStability() {
    Spec               spec;
    std::vector<float> weights(static_cast<std::size_t>(spec.paramCount()), 0.0f);
    for (float& w : weights)
        w = 1000.0f;

    Adapter adapter(spec, weights, NetHash, "test.nnue", "test", {});
    auto    acc = deterministicAccumulator();

    const float delta = adapter.correction(acc.data());
    CHECK(std::isfinite(delta));
    CHECK(std::abs(delta) <= static_cast<float>(spec.limitCp) + 1e-3f);

    std::array<std::int16_t, Stockfish::Adapt::kAccumulatorDimensions> extreme{};
    extreme.fill(std::numeric_limits<std::int16_t>::max());
    const float extremeDelta = adapter.correction(extreme.data());
    CHECK(std::isfinite(extremeDelta));
    CHECK(std::abs(extremeDelta) <= static_cast<float>(spec.limitCp) + 1e-3f);
}

void testProjection() {
    Spec               spec;
    std::vector<float> weights = zeroWeights(spec);
    Adapter            adapter(spec, weights, NetHash, "test.nnue", "test", {});
    auto               acc = deterministicAccumulator();

    std::array<float, Stockfish::Adapt::kMaxInputs> first{};
    std::array<float, Stockfish::Adapt::kMaxInputs> second{};
    adapter.project(acc.data(), first.data());
    adapter.project(acc.data(), second.data());

    for (int i = 0; i < spec.inputDim; ++i)
    {
        CHECK(first[i] == second[i]);
        CHECK(std::isfinite(first[i]));
        CHECK(std::abs(first[i]) <= 32.0f + 1e-4f);
    }
}

void testSerialization() {
    Spec               spec;
    std::vector<float> weights(static_cast<std::size_t>(spec.paramCount()), 0.0f);
    for (int i = 0; i < spec.paramCount(); ++i)
        weights[static_cast<std::size_t>(i)] = std::sin(static_cast<float>(i)) * 0.1f;

    Adapter adapter(spec, weights, NetHash, "test.nnue", "Stockfish test",
                    {{"meta.opponent", "baseline-nodes-2000"}, {"meta.seed", "42"}});

    const std::string path = tempPath();
    std::string       error;
    CHECK(adapter.save(path, error));

    auto loaded = Adapter::load(path, NetHash, error);
    CHECK(loaded != nullptr);
    if (loaded)
    {
        CHECK(loaded->spec().inputDim == spec.inputDim);
        CHECK(loaded->spec().hiddenDim == spec.hiddenDim);
        CHECK(loaded->spec().limitCp == spec.limitCp);
        CHECK(loaded->spec().paramCount() == spec.paramCount());
        CHECK(loaded->netHash() == NetHash);
        CHECK(loaded->netName() == "test.nnue");
        CHECK(loaded->engineId() == "Stockfish test");
        CHECK(loaded->extraHeader().size() == 2);
        CHECK(loaded->weights().size() == weights.size());

        auto acc = deterministicAccumulator();
        CHECK(loaded->correction(acc.data()) == adapter.correction(acc.data()));
    }

    error.clear();
    auto wrongNet = Adapter::load(path, "ffffffffffffffff", error);
    CHECK(wrongNet == nullptr);
    CHECK(!error.empty());

    error.clear();
    auto noHash = Adapter::load(path, "", error);
    CHECK(noHash == nullptr);
    CHECK(!error.empty());

    std::remove(path.c_str());
}

void testRejectsMalformed() {
    const std::string path = tempPath();

    {
        std::ofstream out(path);
        out << "NOTSFA\n";
    }
    std::string error;
    CHECK(Adapter::load(path, NetHash, error) == nullptr);

    Spec               spec;
    std::vector<float> weights(static_cast<std::size_t>(spec.paramCount()), 0.0f);
    Adapter            adapter(spec, weights, NetHash, "test.nnue", "test", {});
    CHECK(adapter.save(path, error));

    {
        std::ifstream in(path);
        std::string   text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto          pos = text.find("feature_version 1");
        if (pos != std::string::npos)
            text.replace(pos, std::string("feature_version 1").size(),
                         "feature_version 999");
        std::ofstream out(path);
        out << text;
    }
    CHECK(Adapter::load(path, NetHash, error) == nullptr);

    std::remove(path.c_str());
}

void testRejectsNonFinite() {
    const std::string path = tempPath();
    Spec              spec;
    std::vector<float> weights(static_cast<std::size_t>(spec.paramCount()), 0.0f);
    Adapter            adapter(spec, weights, NetHash, "test.nnue", "test", {});
    std::string        error;
    CHECK(adapter.save(path, error));

    {
        std::ifstream in(path);
        std::string   text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto          pos = text.find("weights\n");
        if (pos != std::string::npos)
        {
            pos += std::string("weights\n").size();
            auto end = text.find_first_of(" \n", pos);
            text.replace(pos, end - pos, "nan");
        }
        std::ofstream out(path);
        out << text;
    }
    CHECK(Adapter::load(path, NetHash, error) == nullptr);

    std::remove(path.c_str());
}

}  // namespace

int main() {
    testParameterBudget();
    testZeroInitialisation();
    testBoundsAndStability();
    testProjection();
    testSerialization();
    testRejectsMalformed();
    testRejectsNonFinite();

    if (failures != 0)
    {
        std::fprintf(stderr, "%d adapter test(s) failed\n", failures);
        return 1;
    }

    std::printf("adapter tests passed\n");
    return 0;
}
