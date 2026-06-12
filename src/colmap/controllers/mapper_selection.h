// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <string>

namespace colmap {

// Which SfM mapper to run. Mirrors the AutomaticReconstructionController Mapper
// enum (INCREMENTAL | GLOBAL | HIERARCHICAL) without depending on it.
enum class RecommendedMapper {
  kIncremental,
  kGlobal,
  kHierarchical,
};

const char* RecommendedMapperToString(RecommendedMapper mapper);

// Inputs the selector can derive cheaply from the database and environment
// before running any reconstruction.
struct MapperSelectionStats {
  // Number of images in the database.
  int num_images = 0;
  // Fraction of possible image pairs that have a verified two-view geometry,
  // in [0, 1]. Negative means unknown (selector then assumes a typical graph).
  double view_graph_density = -1.0;
  // True if the capture is sequential (video / ordered frames), which favors
  // incremental mapping (weak loop closure degrades global methods).
  bool is_video = false;
};

struct MapperRecommendation {
  RecommendedMapper mapper = RecommendedMapper::kIncremental;
  // Human-readable explanation of the choice (surfaced to the user / orchestrator).
  std::string rationale;
};

// Recommends a mapper from cheap pre-reconstruction signals.
//
// IMPORTANT: the thresholds below are an evidence-informed *heuristic*, not yet
// calibrated by the on-device benchmark suite (todo.md M3a). The decision is
// always surfaced with a rationale so the user can override. The default
// remains incremental (COLMAP's robust, accurate baseline) whenever the inputs
// are ambiguous.
MapperRecommendation SelectMapper(const MapperSelectionStats& stats);

}  // namespace colmap
