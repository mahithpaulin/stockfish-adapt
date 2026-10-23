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

#ifndef ADAPT_INTEGRATION_H_INCLUDED
#define ADAPT_INTEGRATION_H_INCLUDED

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "../types.h"
#include "adapter.h"

namespace Stockfish::Adapt {

// Adds the adapter correction to a raw Stockfish evaluation. Both values are
// side-to-move relative, in Stockfish internal units. The correction is bounded
// by the checkpoint limit and the result never enters the tablebase range.
inline Value applyCorrection(const Adapter& adapter, Value raw, const std::int16_t* accumulator) {
    const int limit = static_cast<int>(std::lround(adapter.spec().limitCp));
    int       delta = static_cast<int>(std::lround(adapter.correction(accumulator)));

    delta = std::clamp(delta, -limit, limit);
    return static_cast<Value>(std::clamp<int>(raw + delta, VALUE_TB_LOSS_IN_MAX_PLY + 1,
                                              VALUE_TB_WIN_IN_MAX_PLY - 1));
}

}  // namespace Stockfish::Adapt

#endif  // ADAPT_INTEGRATION_H_INCLUDED
