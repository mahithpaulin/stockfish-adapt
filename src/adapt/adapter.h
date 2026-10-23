/*
  Stockfish Adapt, a Stockfish variant with a frozen-NNUE evaluation adapter.

  Copyright (C) 2026 the Stockfish Adapt contributors
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ADAPT_ADAPTER_H_INCLUDED
#define ADAPT_ADAPTER_H_INCLUDED

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Stockfish::Adapt {

// Verified against src/nnue/nnue_architecture.h (L1 = 1024). The integration
// point contains a static_assert so a future network change fails to compile
// instead of silently reading the wrong feature vector.
constexpr int kAccumulatorDimensions = 1024;

// Checkpoint format and feature-extraction contract.
constexpr int kFormatVersion    = 1;
constexpr int kFeatureVersion   = 1;
constexpr int kDefaultInputs    = 96;
constexpr int kDefaultHidden    = 20;
constexpr int kLargeInputs      = 125;
constexpr int kLargeHidden      = 32;
constexpr int kMaxInputs        = 192;
constexpr int kMaxHidden        = 64;
constexpr int kMaxParams        = 4096;
constexpr int kProjectionTaps   = 16;
constexpr float kAccumulatorDiv = 512.0f;
constexpr float kFeatureClip    = 8.0f;

enum class Mode { Off, Adapt, Learn };
const char* modeName(Mode mode);

struct Spec {
    int    inputDim  = kDefaultInputs;
    int    hiddenDim = kDefaultHidden;
    double limitCp   = 260.0;

    int paramCount() const { return inputDim * hiddenDim + hiddenDim + hiddenDim + 1; }

    bool valid() const {
        return inputDim >= 1 && inputDim <= kMaxInputs && hiddenDim >= 1
            && hiddenDim <= kMaxHidden && paramCount() <= kMaxParams && limitCp > 0.0
            && limitCp <= 1000.0;
    }
};

// Immutable after construction. All weights are float32.
//
// Weight order in the flat vector:
//   W1: inputDim * hiddenDim, row-major [input][hidden]
//   b1: hiddenDim
//   W2: hiddenDim
//   b2: 1
class Adapter {
   public:
    Adapter(Spec spec,
            std::vector<float> weights,
            std::string netHash,
            std::string netName,
            std::string engineId,
            std::vector<std::pair<std::string, std::string>> extraHeader = {});

    const Spec& spec() const { return spec_; }
    const std::vector<float>& weights() const { return weights_; }
    const std::string& netHash() const { return netHash_; }
    const std::string& netName() const { return netName_; }
    const std::string& engineId() const { return engineId_; }
    const std::vector<std::pair<std::string, std::string>>& extraHeader() const {
        return extraHeader_;
    }

    // Projected, normalised adapter input (length spec().inputDim).
    void project(const std::int16_t* accumulator, float* output) const;

    // Correction in Stockfish internal units (1 pawn = 208 internal units),
    // bounded to (-limitCp, +limitCp). Returns exactly 0 when W2 and b2 are 0.
    float correction(const std::int16_t* accumulator) const;

    bool save(const std::string& path, std::string& error) const;

    // Loads and validates a checkpoint against the current frozen network hash.
    static std::shared_ptr<const Adapter> load(const std::string& path,
                                               const std::string& expectedNetHash,
                                               std::string&       error);

   private:
    Spec                                              spec_;
    std::vector<float>                                weights_;
    std::string                                       netHash_;
    std::string                                       netName_;
    std::string                                       engineId_;
    std::vector<std::pair<std::string, std::string>>  extraHeader_;
};

// Fixed feature-version-1 projection, usable without a loaded adapter.
void projectFeatures(const std::int16_t* accumulator, int inputDim, float* output);

// Process-wide installation. The eval path reads an immutable pointer; loading
// builds a new object and retains every previous object until process exit so
// concurrent evaluators can never observe a freed adapter.
bool install(const std::string& path, const std::string& expectedNetHash, std::string& error);
void unload();
bool setMode(Mode mode, std::string& error);
Mode mode();
const Adapter* active();
// Loaded adapter regardless of mode, for offline feature extraction and probes.
const Adapter* loaded();
void onNetworkChanged(const std::string& newNetHash);
std::string status();

}  // namespace Stockfish::Adapt

#endif  // ADAPT_ADAPTER_H_INCLUDED
