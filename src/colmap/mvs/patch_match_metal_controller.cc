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

// Metal MVS controller: the macOS analogue of mvs::PatchMatchController (which
// is CUDA-only). Reads a COLMAP dense workspace, runs the validated Metal
// PatchMatch optimizer per reference image (photometric pass), then a
// geometric-consistency filtering pass, and writes the depth/normal maps.
//
// This is plain C++ orchestration over the Metal kernels in patch_match_metal.h
// (no Metal/Objective-C here); it is compiled only when COLMAP_METAL_ENABLED.
// Reference/source problems come from stereo/patch-match.cfg when present
// (honoring __all__ / __auto__,N / explicit source lists, see BuildProblems);
// otherwise every image is a reference with most-overlapping auto sources.

#include "colmap/mvs/patch_match_metal.h"

#if defined(COLMAP_METAL_ENABLED)

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/model.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/patch_match_options.h"
#include "colmap/mvs/workspace.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/progress.h"
#include "colmap/util/string.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

namespace colmap {
namespace mvs {
namespace {

// Per-pixel grayscale in [0, 1] from a workspace bitmap (loaded single-channel
// via image_as_rgb=false). Fails loudly if a multi-channel buffer comes back,
// rather than silently misreading the first 1/3 of an RGB buffer.
std::vector<float> GrayFloat(Workspace& workspace, int idx, int& width,
                             int& height) {
  const Bitmap& bitmap = workspace.GetBitmap(idx);
  width = bitmap.Width();
  height = bitmap.Height();
  THROW_CHECK_EQ(bitmap.Channels(), 1)
      << "Metal MVS expects single-channel bitmaps (image_as_rgb=false).";
  const std::vector<uint8_t>& data = bitmap.RowMajorData();
  std::vector<float> gray(static_cast<size_t>(width) * height);
  for (size_t i = 0; i < gray.size(); ++i) gray[i] = data[i] / 255.0f;
  return gray;
}

// Read a previously-written depth map's raw row-major values.
std::vector<float> ReadDepth(const std::filesystem::path& path) {
  DepthMap depth_map;
  depth_map.Read(path);
  return depth_map.GetData();
}

// (reference, source indices) problems. Honors stereo/patch-match.cfg if present
// -- mirroring PatchMatchController::ReadProblems' __all__ / __auto__,N /
// explicit-list semantics -- else processes every image with auto-selected
// sources. Source count is capped at `max_src` (best-K aggregation over the top
// few overlapping views is sufficient and keeps the optimizer fast even when the
// cfg requests many, e.g. "__auto__, 20").
std::vector<std::pair<int, std::vector<int>>> BuildProblems(
    const std::filesystem::path& workspace_path, const Model& model,
    const std::vector<std::vector<int>>& auto_overlap, int max_src) {
  const int num_images = static_cast<int>(model.images.size());
  auto cap = [&](std::vector<int> v) {
    if (static_cast<int>(v.size()) > max_src) v.resize(max_src);
    return v;
  };
  std::vector<std::pair<int, std::vector<int>>> problems;
  const std::filesystem::path cfg_path =
      workspace_path / "stereo" / "patch-match.cfg";
  if (!ExistsFile(cfg_path)) {
    for (int i = 0; i < num_images; ++i)
      problems.emplace_back(i, cap(auto_overlap[i]));
    return problems;
  }

  // Valid image names, so a stale/typo'd cfg entry is skipped with a warning
  // rather than throwing out of GetImageIdx and aborting the whole run.
  std::unordered_set<std::string> valid_names;
  valid_names.reserve(num_images);
  for (int i = 0; i < num_images; ++i) valid_names.insert(model.GetImageName(i));

  const std::vector<std::string> lines = ReadTextFileLines(cfg_path);
  std::string ref_name;
  bool warned_cap = false;  // warn at most once that sources were capped.
  for (const std::string& raw : lines) {
    std::string line = raw;
    StringTrim(&line);
    if (line.empty() || line[0] == '#') continue;
    if (ref_name.empty()) {
      ref_name = line;
      continue;
    }
    const std::string this_ref = ref_name;
    ref_name.clear();  // consume the pair before any early-continue below.

    if (valid_names.count(this_ref) == 0) {
      LOG(WARNING) << "patch-match.cfg references unknown image '" << this_ref
                   << "'; skipping it.";
      continue;
    }
    const int ref_idx = model.GetImageIdx(this_ref);
    const std::vector<std::string> spec = CSVToVector<std::string>(line);

    std::vector<int> srcs;
    const bool is_all = spec.size() == 1 && spec[0] == "__all__";
    const bool is_auto = !spec.empty() && spec[0] == "__auto__";
    if (is_all || is_auto) {
      // __all__ / __auto__[,N]: pick the most-overlapping views. We cap the
      // source count at max_src for optimizer speed; honor an explicit smaller
      // N, and warn (once) when the cfg requested more than the cap so the
      // divergence from the CUDA path is not silent.
      srcs = auto_overlap[ref_idx];  // already <= max_src.
      int requested = is_all ? num_images - 1 : max_src;
      if (is_auto && spec.size() >= 2) requested = std::atoi(spec[1].c_str());
      if (requested > 0 && requested < static_cast<int>(srcs.size())) {
        srcs.resize(requested);
      } else if (requested > max_src && !warned_cap) {
        LOG(WARNING) << "The Metal patch-match backend caps source views at "
                     << max_src << " for speed; patch-match.cfg requested "
                     << requested << " (results may differ from the CUDA path).";
        warned_cap = true;
      }
    } else {
      // Explicit source names: skip unknown names (warn), and exclude the
      // reference itself + duplicates (a self-source trivially passes the
      // geometric cross-check and would inflate the consistency count).
      std::unordered_set<int> seen;
      for (const std::string& name : spec) {
        if (valid_names.count(name) == 0) {
          LOG(WARNING) << "patch-match.cfg lists unknown source image '" << name
                       << "' for '" << this_ref << "'; skipping it.";
          continue;
        }
        const int src_idx = model.GetImageIdx(name);
        if (src_idx == ref_idx) continue;  // a reference is not its own source.
        if (seen.insert(src_idx).second) srcs.push_back(src_idx);
      }
    }
    problems.emplace_back(ref_idx, cap(std::move(srcs)));
  }
  return problems;
}

}  // namespace

void RunPatchMatchStereoMetal(const PatchMatchOptions& options,
                              const std::filesystem::path& workspace_path,
                              const std::string& workspace_format) {
  Workspace::Options wopts;
  wopts.workspace_path = workspace_path.string();
  wopts.workspace_format = workspace_format;
  wopts.input_type = "photometric";
  wopts.image_as_rgb = false;
  wopts.max_image_size = options.max_image_size;
  wopts.cache_size = options.cache_size;
  CachedWorkspace workspace(wopts);
  const Model& model = workspace.GetModel();
  const int num_images = static_cast<int>(model.images.size());
  THROW_CHECK_GT(num_images, 0);

  const int num_src = 6;  // source views per reference (top overlapping).
  const auto overlap =
      model.GetMaxOverlappingImages(num_src, options.min_triangulation_angle);
  const auto ranges = model.ComputeDepthRanges();
  const float sigma_spatial = options.sigma_spatial > 0
                                  ? static_cast<float>(options.sigma_spatial)
                                  : static_cast<float>(options.window_radius);

  const std::filesystem::path depth_dir =
      workspace_path / "stereo" / "depth_maps";
  const std::filesystem::path normal_dir =
      workspace_path / "stereo" / "normal_maps";
  CreateDirIfNotExists(depth_dir);
  CreateDirIfNotExists(normal_dir);

  auto photometric_depth_path = [&](int i) {
    return depth_dir / (model.GetImageName(i) + ".photometric.bin");
  };
  auto photometric_normal_path = [&](int i) {
    return normal_dir / (model.GetImageName(i) + ".photometric.bin");
  };

  std::vector<int> widths(num_images, 0), heights(num_images, 0);
  std::vector<float> depth_min(num_images, 0), depth_max(num_images, 0);

  // Reference/source problems (cfg-driven, else all-images + auto sources).
  const auto problems = BuildProblems(workspace_path, model, overlap, num_src);

  // Sources used to estimate each image's photometric depth: a reference uses
  // its problem's sources; an image needed only as another reference's source
  // uses auto-selection. Only images in `need` are computed in pass 1.
  std::vector<std::vector<int>> photo_src(num_images);
  std::vector<char> is_ref(num_images, 0), need(num_images, 0);
  for (const auto& problem : problems) {
    is_ref[problem.first] = 1;
    need[problem.first] = 1;
    photo_src[problem.first] = problem.second;
    for (int s : problem.second) need[s] = 1;
  }
  for (int i = 0; i < num_images; ++i) {
    if (need[i] && !is_ref[i]) photo_src[i] = overlap[i];
  }

  // Pass 1 (photometric): estimate each image's depth + normal map and write
  // both to disk (.photometric.bin). Writing to disk -- rather than holding all
  // maps in RAM -- bounds peak depth-map memory to O(num_src) in pass 2 and
  // produces the photometric maps downstream stereo_fusion can consume.
  // Liveness during this long (~seconds/image) dense phase; no-op unless
  // --progress_format is set.
  HeartbeatThrottle photometric_heartbeat("patch_match_stereo");
  for (int i = 0; i < num_images; ++i) {
    if (!need[i]) continue;
    photometric_heartbeat.Tick();
    int width = 0, height = 0;
    std::vector<float> ref_gray = GrayFloat(workspace, i, width, height);
    std::vector<float> src_gray, src_k, src_r, src_t;
    int ns = 0;
    for (int s : photo_src[i]) {
      int sw = 0, sh = 0;
      std::vector<float> gray = GrayFloat(workspace, s, sw, sh);
      if (sw != width || sh != height) continue;
      src_gray.insert(src_gray.end(), gray.begin(), gray.end());
      src_k.insert(src_k.end(), model.images[s].GetK(),
                   model.images[s].GetK() + 9);
      src_r.insert(src_r.end(), model.images[s].GetR(),
                   model.images[s].GetR() + 9);
      src_t.insert(src_t.end(), model.images[s].GetT(),
                   model.images[s].GetT() + 3);
      ++ns;
    }
    widths[i] = width;
    heights[i] = height;
    depth_min[i] = options.depth_min > 0 ? static_cast<float>(options.depth_min)
                                         : ranges[i].first;
    depth_max[i] = options.depth_max > 0 ? static_cast<float>(options.depth_max)
                                         : ranges[i].second;

    DepthMap depth_map(width, height, depth_min[i], depth_max[i]);
    NormalMap normal_map(width, height);
    if (ns == 0 || depth_max[i] <= depth_min[i]) {
      LOG(WARNING) << "Skipping " << model.GetImageName(i)
                   << " (no usable source views / depth range)";
    } else {
      const size_t n = static_cast<size_t>(width) * height;
      std::vector<float> depth(n), normal(n * 3);
      const Image& ref_image = model.images[i];
      // filter_min_ncc=0: the geometric pass needs complete source depth maps;
      // photometrically pre-zeroing them would make cross-checks over-reject.
      const bool ok = EstimateDepthMapMetalFromArrays(
          width, height, ref_gray.data(), ref_image.GetK(), ref_image.GetR(),
          ref_image.GetT(), ns, src_gray.data(), src_k.data(), src_r.data(),
          src_t.data(), (ns + 1) / 2, options.window_radius,
          options.window_step, sigma_spatial,
          static_cast<float>(options.sigma_color), depth_min[i], depth_max[i],
          options.num_iterations, /*filter_min_ncc=*/0.0f, depth.data(),
          normal.data());
      THROW_CHECK(ok) << "Metal depth estimation failed for "
                      << model.GetImageName(i);
      for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
          depth_map.Set(r, c, depth[static_cast<size_t>(r) * width + c]);
          for (int k = 0; k < 3; ++k) {
            normal_map.Set(
                r, c, k,
                normal[(static_cast<size_t>(r) * width + c) * 3 + k]);
          }
        }
      }
    }
    depth_map.Write(photometric_depth_path(i));
    normal_map.Write(photometric_normal_path(i));
    LOG(INFO) << StringPrintf("Photometric depth %d/%d: %s (%dx%d, %d src)",
                              i + 1, num_images, model.GetImageName(i).c_str(),
                              width, height, ns);
  }

  // Pass 2 (geometric): stream back the reference + its source depth maps, keep
  // pixels consistent in >= filter_min_num_consistent source views (COLMAP's
  // rule; averaging would let one occluded view reject a good pixel), and write
  // the filtered geometric depth + normal maps.
  for (const auto& problem : problems) {
    const int i = problem.first;
    const std::vector<int>& srcs = problem.second;
    const int width = widths[i], height = heights[i];
    if (width == 0 || height == 0) continue;
    const size_t n = static_cast<size_t>(width) * height;
    std::vector<float> depth = ReadDepth(photometric_depth_path(i));
    if (depth.size() != n) continue;

    if (options.geom_consistency) {
      const float* k9 = model.images[i].GetK();
      const float ref_inv_k[4] = {1.0f / k9[0], -k9[2] / k9[0], 1.0f / k9[4],
                                  -k9[5] / k9[4]};
      const float ref_k[4] = {k9[0], k9[2], k9[4], k9[5]};
      std::vector<int> rows(n), cols(n);
      for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
          rows[static_cast<size_t>(r) * width + c] = r;
          cols[static_cast<size_t>(r) * width + c] = c;
        }
      }
      // A pixel survives iff its depth is geometrically consistent (forward-
      // backward reprojection error <= keep_max) in at least
      // filter_min_num_consistent source views -- COLMAP's count-based rule;
      // averaging the cost over sources would let one occluded view reject an
      // otherwise good pixel.
      const float keep_max =
          static_cast<float>(options.filter_geom_consistency_max_cost);

      // Gather every valid source's projection matrices and depth map into
      // contiguous buffers, then evaluate all sources in ONE fused dispatch that
      // returns the per-pixel consistent-source count (fewer GPU dispatches and
      // a single source-depth upload vs one dispatch per source).
      std::vector<float> src_P, src_inv_P, src_depths;
      int ns = 0;
      for (int s : srcs) {
        if (widths[s] != width || heights[s] != height) continue;
        std::vector<float> src_depth = ReadDepth(photometric_depth_path(s));
        if (src_depth.size() != n) continue;
        float r_rel[9], t_rel[3], P[12], inv_P[12];
        ComputeRelativePose(model.images[i].GetR(), model.images[i].GetT(),
                            model.images[s].GetR(), model.images[s].GetT(),
                            r_rel, t_rel);
        ComposeProjectionMatrix(model.images[s].GetK(), r_rel, t_rel, P);
        ComposeInverseProjectionMatrix(model.images[s].GetK(), r_rel, t_rel,
                                       inv_P);
        src_P.insert(src_P.end(), P, P + 12);
        src_inv_P.insert(src_inv_P.end(), inv_P, inv_P + 12);
        src_depths.insert(src_depths.end(), src_depth.begin(), src_depth.end());
        ++ns;
      }
      if (ns == 0) {
        // No same-resolution source could be loaded, so this reference cannot
        // be geometrically cross-checked. Honor the geom_consistency contract by
        // dropping its (unverified) depths rather than writing them through as a
        // "geometric" map that would pollute stereo_fusion.
        LOG(WARNING) << "No usable source views for geometric consistency of "
                     << model.GetImageName(i) << "; dropping its depths.";
        std::fill(depth.begin(), depth.end(), 0.0f);
      } else {
        // Fail loudly on a Metal dispatch error rather than silently writing
        // unfiltered photometric depth as the geometric output.
        std::vector<int> consistent(n, 0);
        THROW_CHECK(ComputeGeomConsistencyCountMetal(
            width, height, ref_inv_k, ref_k, src_P.data(), src_inv_P.data(),
            src_depths.data(), ns, keep_max, rows.data(), cols.data(),
            depth.data(), static_cast<int>(n), consistent.data()))
            << "Metal geometric-consistency dispatch failed for "
            << model.GetImageName(i);
        const int min_consistent =
            std::min(options.filter_min_num_consistent, ns);
        for (size_t p = 0; p < n; ++p) {
          if (consistent[p] < min_consistent) depth[p] = 0.0f;
        }
      }
    }

    DepthMap depth_map(width, height, depth_min[i], depth_max[i]);
    for (int r = 0; r < height; ++r) {
      for (int c = 0; c < width; ++c) {
        depth_map.Set(r, c, depth[static_cast<size_t>(r) * width + c]);
      }
    }
    depth_map.Write(depth_dir / (model.GetImageName(i) + ".geometric.bin"));

    // Geometric normal = photometric normal masked to the surviving depths.
    NormalMap normal_map;
    normal_map.Read(photometric_normal_path(i));
    const std::vector<float>& nd = normal_map.GetData();
    if (nd.size() == n * 3) {
      NormalMap geo_normal(width, height);
      for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
          const size_t p = static_cast<size_t>(r) * width + c;
          if (depth[p] != 0.0f) {
            for (int k = 0; k < 3; ++k) {
              geo_normal.Set(r, c, k, nd[static_cast<size_t>(k) * n + p]);
            }
          }
        }
      }
      geo_normal.Write(normal_dir / (model.GetImageName(i) + ".geometric.bin"));
    }
  }
  LOG(INFO) << "Metal patch-match stereo complete: wrote " << problems.size()
            << " geometric depth/normal maps to "
            << (workspace_path / "stereo").string();
}

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_METAL_ENABLED
