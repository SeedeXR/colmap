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

#include "colmap/controllers/mapper_selection.h"

#include "colmap/util/string.h"

namespace colmap {

double ComputeViewGraphDensity(int num_images, size_t num_verified_pairs) {
  if (num_images < 2) {
    return -1.0;
  }
  const double max_pairs =
      0.5 * static_cast<double>(num_images) * (num_images - 1);
  if (max_pairs <= 0) {
    return -1.0;
  }
  return static_cast<double>(num_verified_pairs) / max_pairs;
}

const char* RecommendedMapperToString(RecommendedMapper mapper) {
  switch (mapper) {
    case RecommendedMapper::kIncremental: return "incremental";
    case RecommendedMapper::kGlobal: return "global";
    case RecommendedMapper::kHierarchical: return "hierarchical";
  }
  return "incremental";
}

namespace {
// Heuristic thresholds (todo.md M3a: to be calibrated by on-device benchmarks;
// chosen conservatively so the robust incremental default is preferred unless
// there is a clear reason to switch).
constexpr int kSmallSetMaxImages = 200;       // <= this -> incremental.
constexpr int kHierarchicalMinImages = 2000;  // >= this -> hierarchical.
constexpr double kWellConnectedDensity = 0.10;  // dense graph -> global viable.
}  // namespace

MapperRecommendation SelectMapper(const MapperSelectionStats& stats) {
  MapperRecommendation rec;

  // Sequential capture: incremental handles weak loop closure best; global
  // rotation averaging degrades on chain-like graphs.
  if (stats.is_video) {
    rec.mapper = RecommendedMapper::kIncremental;
    rec.rationale =
        "sequential/video capture favors incremental mapping (global methods "
        "need a well-conditioned view graph)";
    return rec;
  }

  // Very large collections: cluster -> map -> merge.
  if (stats.num_images >= kHierarchicalMinImages) {
    rec.mapper = RecommendedMapper::kHierarchical;
    rec.rationale = StringPrintf(
        "%d images exceeds the single-pass threshold (%d); hierarchical "
        "cluster/map/merge scales best",
        stats.num_images,
        kHierarchicalMinImages);
    return rec;
  }

  // Small/medium sets: incremental is the robust, accurate baseline and the
  // speed advantage of global SfM is small at this scale.
  if (stats.num_images <= kSmallSetMaxImages) {
    rec.mapper = RecommendedMapper::kIncremental;
    rec.rationale = StringPrintf(
        "%d images is small enough that incremental mapping is the most "
        "robust choice with negligible speed penalty",
        stats.num_images);
    return rec;
  }

  // Medium-large, unordered: global (GLOMAP) is faster on well-connected
  // graphs; fall back to incremental when the graph is sparse or unknown.
  if (stats.view_graph_density >= kWellConnectedDensity) {
    rec.mapper = RecommendedMapper::kGlobal;
    rec.rationale = StringPrintf(
        "%d images with a well-connected view graph (density %.2f >= %.2f); "
        "global SfM (GLOMAP) is typically faster at comparable quality",
        stats.num_images,
        stats.view_graph_density,
        kWellConnectedDensity);
    return rec;
  }

  rec.mapper = RecommendedMapper::kIncremental;
  if (stats.view_graph_density < 0.0) {
    rec.rationale = StringPrintf(
        "%d images but view-graph connectivity is unknown; defaulting to the "
        "robust incremental mapper",
        stats.num_images);
  } else {
    rec.rationale = StringPrintf(
        "%d images with a sparse view graph (density %.2f < %.2f); incremental "
        "mapping is more robust than global on weakly connected graphs",
        stats.num_images,
        stats.view_graph_density,
        kWellConnectedDensity);
  }
  return rec;
}

}  // namespace colmap
