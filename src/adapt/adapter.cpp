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

#include "adapter.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>

namespace Stockfish::Adapt {

namespace {

struct Tap {
    std::uint32_t index;
    float         sign;
};

using Projection = std::array<Tap, kMaxInputs * kProjectionTaps>;

std::uint64_t splitmix64(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Feature version 1 projection. The seed is part of the checkpoint contract:
// changing it requires a new feature version.
const Projection& projection() {
    static const Projection table = [] {
        Projection result{};
        std::uint64_t state = 0x53544F434B464953ULL;
        for (int r = 0; r < kMaxInputs; ++r)
            for (int t = 0; t < kProjectionTaps; ++t)
            {
                const std::uint64_t z = splitmix64(state);
                result[static_cast<std::size_t>(r) * kProjectionTaps + t] = {
                  static_cast<std::uint32_t>(z % kAccumulatorDimensions),
                  (z >> 63) ? -1.0f : 1.0f};
            }
        return result;
    }();
    return table;
}

std::string trim(std::string s) {
    const auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
    return s;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool parseInt(const std::string& text, long long& value) {
    try
    {
        std::size_t used = 0;
        value            = std::stoll(text, &used);
        return used == text.size();
    }
    catch (...)
    {
        return false;
    }
}

bool parseDouble(const std::string& text, double& value) {
    try
    {
        std::size_t used = 0;
        value            = std::stod(text, &used);
        return used == text.size();
    }
    catch (...)
    {
        return false;
    }
}

bool sameNetHash(const std::string& a, const std::string& b) {
    std::string left  = lower(a);
    std::string right = lower(b);
    if (left.rfind("0x", 0) == 0)
        left.erase(0, 2);
    if (right.rfind("0x", 0) == 0)
        right.erase(0, 2);
    return !left.empty() && left == right;
}

}  // namespace

Adapter::Adapter(Spec spec,
                 std::vector<float> weights,
                 std::string netHash,
                 std::string netName,
                 std::string engineId,
                 std::vector<std::pair<std::string, std::string>> extraHeader) :
    spec_(spec),
    weights_(std::move(weights)),
    netHash_(std::move(netHash)),
    netName_(std::move(netName)),
    engineId_(std::move(engineId)),
    extraHeader_(std::move(extraHeader)) {}

void projectFeatures(const std::int16_t* accumulator, int inputDim, float* output) {
    const Projection& table = projection();
    inputDim = std::clamp(inputDim, 1, kMaxInputs);

    for (int r = 0; r < inputDim; ++r)
    {
        float       sum = 0.0f;
        const Tap*  row = &table[static_cast<std::size_t>(r) * kProjectionTaps];
        for (int t = 0; t < kProjectionTaps; ++t)
        {
            const float x = std::clamp(static_cast<float>(accumulator[row[t].index])
                                         * (1.0f / kAccumulatorDiv),
                                       -kFeatureClip, kFeatureClip);
            sum += row[t].sign * x;
        }
        output[r] = sum * 0.25f;
    }
}

void Adapter::project(const std::int16_t* accumulator, float* output) const {
    projectFeatures(accumulator, spec_.inputDim, output);
}

float Adapter::correction(const std::int16_t* accumulator) const {
    const int inputDim  = spec_.inputDim;
    const int hiddenDim = spec_.hiddenDim;

    std::array<float, kMaxInputs> hiddenInput{};
    std::array<float, kMaxHidden> hidden{};

    project(accumulator, hiddenInput.data());

    const float* w1 = weights_.data();
    const float* b1 = w1 + static_cast<std::size_t>(inputDim) * hiddenDim;
    const float* w2 = b1 + hiddenDim;
    const float  b2 = w2[hiddenDim];

    for (int k = 0; k < hiddenDim; ++k)
    {
        float value = b1[k];
        for (int i = 0; i < inputDim; ++i)
            value += w1[static_cast<std::size_t>(i) * hiddenDim + k] * hiddenInput[i];
        hidden[k] = std::tanh(value);
    }

    float value = b2;
    for (int k = 0; k < hiddenDim; ++k)
        value += w2[k] * hidden[k];

    const float delta = static_cast<float>(spec_.limitCp) * std::tanh(value);
    return std::clamp(delta, -static_cast<float>(spec_.limitCp),
                      static_cast<float>(spec_.limitCp));
}

bool Adapter::save(const std::string& path, std::string& error) const {
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        error = "cannot open checkpoint for writing: " + path;
        return false;
    }

    out << "SFA1\n";
    out << "format_version " << kFormatVersion << '\n';
    out << "feature_version " << kFeatureVersion << '\n';
    out << "engine_id " << engineId_ << '\n';
    out << "net_hash " << netHash_ << '\n';
    out << "net_name " << netName_ << '\n';
    out << "arch_input " << spec_.inputDim << '\n';
    out << "arch_hidden " << spec_.hiddenDim << '\n';
    out << "param_count " << spec_.paramCount() << '\n';
    out << "limit_cp " << std::setprecision(9) << spec_.limitCp << '\n';
    for (const auto& [key, value] : extraHeader_)
        out << key << ' ' << value << '\n';
    out << "weights\n";
    out << std::setprecision(9);
    for (std::size_t i = 0; i < weights_.size(); ++i)
    {
        out << weights_[i];
        out << ((i + 1) % 8 == 0 ? '\n' : ' ');
    }
    out << "\nEND\n";

    if (!out)
    {
        error = "write failed: " + path;
        return false;
    }
    return true;
}

std::shared_ptr<const Adapter> Adapter::load(const std::string& path,
                                             const std::string& expectedNetHash,
                                             std::string&       error) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        error = "cannot open adapter checkpoint: " + path;
        return nullptr;
    }

    std::string line;
    if (!std::getline(in, line) || trim(line) != "SFA1")
    {
        error = "not an SFA1 adapter checkpoint: " + path;
        return nullptr;
    }

    std::map<std::string, std::string> header;
    bool                               sawWeights = false;
    while (std::getline(in, line))
    {
        line = trim(line);
        if (line.empty())
            continue;
        if (line == "weights")
        {
            sawWeights = true;
            break;
        }

        const auto split = line.find_first_of(" \t");
        const std::string key = split == std::string::npos ? line : line.substr(0, split);
        const std::string value =
          split == std::string::npos ? std::string{} : trim(line.substr(split + 1));

        if (header.count(key))
        {
            error = "duplicate adapter header key: " + key;
            return nullptr;
        }
        header.emplace(key, value);
    }

    if (!sawWeights)
    {
        error = "adapter checkpoint has no weights section";
        return nullptr;
    }

    const auto required = [&](const std::string& key, std::string& value) {
        const auto it = header.find(key);
        if (it == header.end())
        {
            error = "missing adapter header: " + key;
            return false;
        }
        value = it->second;
        return true;
    };

    std::string formatVersion, featureVersion, netHash, netName, engineId;
    std::string archInput, archHidden, paramCountText, limitText;
    if (!required("format_version", formatVersion) || !required("feature_version", featureVersion)
        || !required("net_hash", netHash) || !required("arch_input", archInput)
        || !required("arch_hidden", archHidden) || !required("param_count", paramCountText)
        || !required("limit_cp", limitText))
        return nullptr;

    long long format = 0, feature = 0, inputDim = 0, hiddenDim = 0, params = 0;
    double    limit = 0.0;
    if (!parseInt(formatVersion, format) || format != kFormatVersion)
    {
        error = "unsupported adapter format_version: " + formatVersion;
        return nullptr;
    }
    if (!parseInt(featureVersion, feature) || feature != kFeatureVersion)
    {
        error = "incompatible adapter feature_version: " + featureVersion;
        return nullptr;
    }
    if (!parseInt(archInput, inputDim) || !parseInt(archHidden, hiddenDim)
        || !parseInt(paramCountText, params) || !parseDouble(limitText, limit))
    {
        error = "malformed adapter numeric header";
        return nullptr;
    }

    Spec spec;
    spec.inputDim  = static_cast<int>(inputDim);
    spec.hiddenDim = static_cast<int>(hiddenDim);
    spec.limitCp   = limit;

    if (!spec.valid())
    {
        error = "invalid adapter architecture or correction limit";
        return nullptr;
    }
    if (params != spec.paramCount())
    {
        error = "adapter param_count does not match architecture";
        return nullptr;
    }
    if (expectedNetHash.empty())
    {
        error = "cannot validate adapter without a loaded network hash";
        return nullptr;
    }
    if (!sameNetHash(netHash, expectedNetHash))
    {
        error = "adapter was trained for network " + netHash + ", current network is "
              + expectedNetHash;
        return nullptr;
    }

    std::vector<float> weights;
    weights.reserve(static_cast<std::size_t>(params));

    std::string token;
    bool        sawEnd = false;
    while (in >> token)
    {
        if (token == "END")
        {
            sawEnd = true;
            break;
        }

        double value = 0.0;
        if (!parseDouble(token, value) || !std::isfinite(value))
        {
            error = "invalid adapter weight token: " + token;
            return nullptr;
        }
        weights.push_back(static_cast<float>(value));
        if (weights.size() > static_cast<std::size_t>(params))
        {
            error = "adapter contains more weights than declared";
            return nullptr;
        }
    }

    if (!sawEnd)
    {
        error = "adapter checkpoint is missing END";
        return nullptr;
    }
    if (weights.size() != static_cast<std::size_t>(params))
    {
        error = "adapter weight count does not match param_count";
        return nullptr;
    }

    auto it = header.find("net_name");
    if (it != header.end())
        netName = it->second;
    it = header.find("engine_id");
    if (it != header.end())
        engineId = it->second;

    std::vector<std::pair<std::string, std::string>> extras;
    for (const auto& [key, value] : header)
    {
        if (key == "format_version" || key == "feature_version" || key == "engine_id"
            || key == "net_hash" || key == "net_name" || key == "arch_input"
            || key == "arch_hidden" || key == "param_count" || key == "limit_cp")
            continue;
        extras.emplace_back(key, value);
    }

    return std::make_shared<Adapter>(spec, std::move(weights), netHash, netName, engineId,
                                     std::move(extras));
}

namespace {

std::atomic<const Adapter*> g_loaded{nullptr};
std::atomic<int>           g_mode{static_cast<int>(Mode::Off)};
std::mutex                 g_mutex;
std::vector<std::shared_ptr<const Adapter>> g_retained;
std::string                g_path;

}  // namespace

const char* modeName(Mode mode) {
    switch (mode)
    {
    case Mode::Adapt :
        return "ADAPT";
    case Mode::Learn :
        return "LEARN";
    default :
        return "OFF";
    }
}

bool install(const std::string& path, const std::string& expectedNetHash, std::string& error) {
    auto adapter = Adapter::load(path, expectedNetHash, error);
    if (!adapter)
        return false;

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_retained.push_back(adapter);
        g_path = path;
    }
    g_loaded.store(adapter.get(), std::memory_order_release);
    return true;
}

void unload() {
    g_mode.store(static_cast<int>(Mode::Off), std::memory_order_release);
    g_loaded.store(nullptr, std::memory_order_release);
    std::lock_guard<std::mutex> lock(g_mutex);
    g_path.clear();
}

bool setMode(Mode mode, std::string& error) {
    if (mode != Mode::Off && g_loaded.load(std::memory_order_acquire) == nullptr)
    {
        error = "no adapter checkpoint loaded; set AdaptFile first";
        return false;
    }
    g_mode.store(static_cast<int>(mode), std::memory_order_release);
    return true;
}

Mode mode() {
    return static_cast<Mode>(g_mode.load(std::memory_order_acquire));
}

const Adapter* active() {
    if (mode() == Mode::Off)
        return nullptr;
    return g_loaded.load(std::memory_order_acquire);
}

const Adapter* loaded() { return g_loaded.load(std::memory_order_acquire); }

void onNetworkChanged(const std::string& newNetHash) {
    const Adapter* current = g_loaded.load(std::memory_order_acquire);
    if (current == nullptr)
        return;

    if (!sameNetHash(current->netHash(), newNetHash))
    {
        g_mode.store(static_cast<int>(Mode::Off), std::memory_order_release);
        g_loaded.store(nullptr, std::memory_order_release);
        std::lock_guard<std::mutex> lock(g_mutex);
        g_path.clear();
    }
}

std::string status() {
    const Adapter* current = g_loaded.load(std::memory_order_acquire);
    std::ostringstream ss;
    ss << "mode=" << modeName(mode());
    if (current == nullptr)
    {
        ss << " adapter=none";
        return ss.str();
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    ss << " adapter=" << g_path << " params=" << current->spec().paramCount()
       << " input=" << current->spec().inputDim << " hidden=" << current->spec().hiddenDim
       << " limit_cp=" << current->spec().limitCp << " feature_version=" << kFeatureVersion
       << " net_hash=" << current->netHash();
    return ss.str();
}

}  // namespace Stockfish::Adapt
