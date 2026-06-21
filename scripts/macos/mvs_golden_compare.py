#!/usr/bin/env python3
# COLMAP dense MVS depth/normal-map validation gate.
#
# Diffs a CANDIDATE `patch_match_stereo` output (e.g. the Metal MVS port) against
# a CUDA GOLDEN reference, on overlapping VALID pixels (depth > 0). This is the
# "did I port the SAME algorithm faithfully?" check that the CUDA golden bundle
# (golden_task/colmap_golden_bundle/) unblocks for the Metal MVS port. For
# absolute accuracy vs ground truth, use ETH3D/DTU instead (see
# memory/testing_mechanism.md §5).
#
# Reference format: COLMAP dense binary map -- ASCII header "W&H&C&" then
# little-endian float32 in Fortran (column-major) order; depth 0 = invalid.
# (Matches COLMAP scripts/python/read_write_dense.py::read_array.)
#
# Usage:
#   scripts/macos/mvs_golden_compare.py --ref REF --cand CAND [options]
#     REF/CAND may point at a `dense/` dir, a `stereo/` dir, or a `depth_maps/`
#     dir -- the script resolves stereo/{depth_maps,normal_maps} under it.
#   --map-type {geometric,photometric}   which *.<type>.bin maps to compare
#   --depth-rel-tol  F   median relative depth error PASS threshold (default 0.01)
#   --normal-deg-tol F   median normal angular error PASS threshold (default 5.0)
#   --min-coverage   F   min fraction of ref-valid pixels the candidate also
#                        covers, PASS threshold (default 0.80)
#   --max-images     N   cap images compared (incremental testing; default all)
#   --json               emit a single-line JSON summary on stdout
#   --selftest           ignore --cand: compare REF against a perturbed copy of
#                        itself to prove the metrics respond (no candidate yet)
#
# Exit 0 = PASS (all gates met), 1 = FAIL, 2 = usage/IO error.
# Memory: streams ONE image pair at a time (never loads the whole set) so it
# stays well within the project RAM budget (memory/testing_mechanism.md).

import argparse
import glob
import json
import os
import sys

import numpy as np


def read_colmap_array(path):
    """Read a COLMAP dense binary map -> HxW (depth) or HxWx3 (normals) float32."""
    with open(path, "rb") as fid:
        header = b""
        amp = 0
        while amp < 3:
            ch = fid.read(1)
            if not ch:
                raise ValueError("truncated header in %s" % path)
            header += ch
            if ch == b"&":
                amp += 1
        width, height, channels = (int(x) for x in header.decode().split("&")[:3])
        data = np.fromfile(fid, dtype=np.float32)
    data = data.reshape((width, height, channels), order="F")
    return np.transpose(data, (1, 0, 2)).squeeze()


def resolve_stereo_dir(path):
    """Accept a dense/, stereo/, or depth_maps/ dir; return the stereo/ dir."""
    if os.path.isdir(os.path.join(path, "stereo", "depth_maps")):
        return os.path.join(path, "stereo")
    if os.path.isdir(os.path.join(path, "depth_maps")):
        return path
    if os.path.basename(os.path.normpath(path)) == "depth_maps":
        return os.path.dirname(os.path.normpath(path))
    raise ValueError("no stereo/depth_maps found under %s" % path)


def map_key(filename, map_type):
    """'P1180141.JPG.geometric.bin' -> 'P1180141.JPG' (strip .<type>.bin)."""
    return filename[: -len(".%s.bin" % map_type)]


def compare_depth(ref, cand):
    """Per-image depth stats on pixels valid (>0) in BOTH maps."""
    if ref.shape != cand.shape:
        return None
    valid = (ref > 0) & (cand > 0)
    ref_valid = int((ref > 0).sum())
    n = int(valid.sum())
    if n == 0 or ref_valid == 0:
        return dict(coverage=0.0, n=0, ref_valid=ref_valid)
    abs_err = np.abs(ref[valid] - cand[valid])
    rel_err = abs_err / np.maximum(ref[valid], 1e-6)
    return dict(
        coverage=n / ref_valid,
        n=n,
        ref_valid=ref_valid,
        abs_median=float(np.median(abs_err)),
        rel_median=float(np.median(rel_err)),
        rel_p95=float(np.percentile(rel_err, 95)),
        rmse=float(np.sqrt(np.mean((ref[valid] - cand[valid]) ** 2))),
        within_1pct=float(np.mean(rel_err < 0.01)),
    )


def compare_normals(ref, cand, ref_depth, cand_depth):
    """Per-image angular error (deg) where depth is valid in both."""
    if ref.shape != cand.shape or ref.ndim != 3:
        return None
    valid = (ref_depth > 0) & (cand_depth > 0)
    if int(valid.sum()) == 0:
        return None
    a = ref[valid]
    b = cand[valid]
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    dots = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    deg = np.degrees(np.arccos(dots))
    return dict(deg_median=float(np.median(deg)), deg_p95=float(np.percentile(deg, 95)))


def run(ref_dir, cand_dir, map_type, max_images):
    """Stream-compare every shared map; return (per_image, aggregate)."""
    ref_depth_dir = os.path.join(ref_dir, "depth_maps")
    cand_depth_dir = os.path.join(cand_dir, "depth_maps")
    suffix = ".%s.bin" % map_type
    ref_files = {
        map_key(f, map_type): f
        for f in os.listdir(ref_depth_dir)
        if f.endswith(suffix)
    }
    cand_files = {
        map_key(f, map_type): f
        for f in os.listdir(cand_depth_dir)
        if f.endswith(suffix)
    }
    shared = sorted(set(ref_files) & set(cand_files))
    if max_images:
        shared = shared[:max_images]
    if not shared:
        raise ValueError(
            "no shared %s depth maps between ref and cand (ref=%d, cand=%d)"
            % (map_type, len(ref_files), len(cand_files))
        )

    rows = []
    for key in shared:
        rd = read_colmap_array(os.path.join(ref_depth_dir, ref_files[key]))
        cd = read_colmap_array(os.path.join(cand_depth_dir, cand_files[key]))
        d = compare_depth(rd, cd)
        row = dict(image=key, depth=d, normal=None)
        rn_path = os.path.join(ref_dir, "normal_maps", key + suffix)
        cn_path = os.path.join(cand_dir, "normal_maps", key + suffix)
        if d and d["n"] > 0 and os.path.exists(rn_path) and os.path.exists(cn_path):
            rn = read_colmap_array(rn_path)
            cn = read_colmap_array(cn_path)
            row["normal"] = compare_normals(rn, cn, rd, cd)
        rows.append(row)

    def agg(getter):
        vals = [getter(r) for r in rows if getter(r) is not None]
        return float(np.mean(vals)) if vals else None

    aggregate = dict(
        images=len(rows),
        coverage=agg(lambda r: r["depth"]["coverage"] if r["depth"] else None),
        depth_rel_median=agg(
            lambda r: r["depth"].get("rel_median") if r["depth"] else None
        ),
        depth_rel_p95=agg(lambda r: r["depth"].get("rel_p95") if r["depth"] else None),
        depth_rmse=agg(lambda r: r["depth"].get("rmse") if r["depth"] else None),
        normal_deg_median=agg(
            lambda r: r["normal"]["deg_median"] if r["normal"] else None
        ),
    )
    return rows, aggregate


def make_perturbed_copy(ref_dir, map_type, rel_noise, max_images):
    """Self-test: write a noised copy of REF's maps to a temp dir, return it."""
    import shutil
    import tempfile

    dst = tempfile.mkdtemp(prefix="mvs_selftest_")
    for sub in ("depth_maps", "normal_maps"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
    suffix = ".%s.bin" % map_type
    files = sorted(
        f for f in os.listdir(os.path.join(ref_dir, "depth_maps")) if f.endswith(suffix)
    )
    if max_images:
        files = files[:max_images]
    rng = np.random.default_rng(0)
    for f in files:
        # Depth: scale valid pixels by (1 + rel_noise), re-emit in COLMAP format.
        src = os.path.join(ref_dir, "depth_maps", f)
        arr = read_colmap_array(src)
        valid = arr > 0
        out = arr.copy()
        out[valid] = arr[valid] * (1.0 + rel_noise)
        _write_colmap_array(os.path.join(dst, "depth_maps", f), out)
        # Normals: copy unchanged so angular error stays ~0 (depth-only perturbation).
        nsrc = os.path.join(ref_dir, "normal_maps", f)
        if os.path.exists(nsrc):
            shutil.copy(nsrc, os.path.join(dst, "normal_maps", f))
    return dst


def _write_colmap_array(path, arr):
    """Write HxW or HxWx3 float32 in COLMAP dense format (column-major)."""
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    with open(path, "wb") as fid:
        fid.write(("%d&%d&%d&" % (w, h, c)).encode())
        np.transpose(arr, (1, 0, 2)).astype(np.float32).ravel(order="F").tofile(fid)


def main():
    ap = argparse.ArgumentParser(description="COLMAP dense MVS validation gate")
    ap.add_argument("--ref", required=True, help="golden reference dir")
    ap.add_argument("--cand", help="candidate (Metal) dir; omit with --selftest")
    ap.add_argument("--map-type", default="geometric", choices=["geometric", "photometric"])
    ap.add_argument("--depth-rel-tol", type=float, default=0.01)
    ap.add_argument("--normal-deg-tol", type=float, default=5.0)
    ap.add_argument("--min-coverage", type=float, default=0.80)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="compare REF vs a perturbed copy of itself")
    ap.add_argument("--selftest-noise", type=float, default=0.05,
                    help="relative depth noise for --selftest (default 0.05 = 5%%)")
    args = ap.parse_args()

    try:
        ref_dir = resolve_stereo_dir(args.ref)
        if args.selftest:
            cand_dir = make_perturbed_copy(
                ref_dir, args.map_type, args.selftest_noise, args.max_images
            )
        else:
            if not args.cand:
                ap.error("--cand is required unless --selftest is given")
            cand_dir = resolve_stereo_dir(args.cand)
        rows, agg = run(ref_dir, cand_dir, args.map_type, args.max_images)
    except (ValueError, OSError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2

    # Gates.
    checks = []
    cov = agg["coverage"]
    checks.append(("coverage >= %.2f" % args.min_coverage,
                   cov is not None and cov >= args.min_coverage, cov))
    drm = agg["depth_rel_median"]
    checks.append(("depth_rel_median <= %.4f" % args.depth_rel_tol,
                   drm is not None and drm <= args.depth_rel_tol, drm))
    ndm = agg["normal_deg_median"]
    if ndm is not None:
        checks.append(("normal_deg_median <= %.2f" % args.normal_deg_tol,
                       ndm <= args.normal_deg_tol, ndm))
    passed = all(ok for _, ok, _ in checks)

    if args.json:
        print(json.dumps(dict(result="PASS" if passed else "FAIL",
                              map_type=args.map_type, **agg)))
    else:
        mode = "SELF-TEST (perturbed %.0f%%)" % (args.selftest_noise * 100) if args.selftest else "compare"
        print("MVS golden %s | map_type=%s | images=%d" % (mode, args.map_type, agg["images"]))
        print("  coverage            = %s" % _fmt(agg["coverage"]))
        print("  depth rel err median= %s   p95 = %s" % (_fmt(agg["depth_rel_median"]), _fmt(agg["depth_rel_p95"])))
        print("  depth RMSE          = %s" % _fmt(agg["depth_rmse"]))
        print("  normal err deg med  = %s" % _fmt(agg["normal_deg_median"]))
        print("  --- gates ---")
        for name, ok, val in checks:
            print("  [%s] %s (got %s)" % ("PASS" if ok else "FAIL", name, _fmt(val)))
        print("RESULT: %s" % ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


def _fmt(v):
    return "n/a" if v is None else ("%.5f" % v)


if __name__ == "__main__":
    sys.exit(main())
