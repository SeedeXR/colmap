#!/bin/bash
# COLMAP macOS performance-baseline tracker.
#
# Complements pipeline_regression.sh: that script enforces ABSOLUTE budgets
# (hard 6 GB RAM cap, sub-pixel reprojection). This one tracks RELATIVE drift
# over time -- it runs the sparse pipeline at a fixed image count, records
# per-stage wall time + peak RSS + quality to a JSON baseline, and on later runs
# DIFFS against that baseline, failing if any metric regresses beyond tolerance.
# Use it to catch "death by a thousand cuts" slowdowns that stay within budget.
#
# Methodology matches memory/testing_mechanism.md (CPU SIFT first_octave=0 + RAM
# budget; CPU matching -- the Metal matcher was removed as slower, see
# memory/future_enhancements.md B). south-building is the canonical baseline set.
#
# Usage:
#   scripts/macos/perf_baseline.sh --update          # write/refresh the baseline
#   scripts/macos/perf_baseline.sh                   # compare vs baseline (gate)
# Options:
#   --bin PATH        colmap binary (default build/src/colmap/exe/colmap)
#   --dataset DIR     dataset dir with images/ (default datasets/south-building)
#   --size N          number of images (default 8)
#   --baseline FILE   baseline JSON path (default scripts/macos/perf_baseline.json)
#   --time-tol PCT    allowed wall-time regression % (default 25)
#   --rss-tol PCT     allowed peak-RSS regression % (default 15)
#   --update          (re)write the baseline instead of comparing
#
# Exit 0 = baseline written, or compare within tolerance. Non-zero = regression
# or setup error.

set -u

BIN="${COLMAP_BIN:-build/src/colmap/exe/colmap}"
DATASET="${COLMAP_DATASET:-datasets/south-building}"
SIZE=8
BASELINE="scripts/macos/perf_baseline.json"
TIME_TOL=25
RSS_TOL=15
UPDATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) BIN="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --size) SIZE="$2"; shift 2;;
    --baseline) BASELINE="$2"; shift 2;;
    --time-tol) TIME_TOL="$2"; shift 2;;
    --rss-tol) RSS_TOL="$2"; shift 2;;
    --update) UPDATE=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

if [[ ! -x "$BIN" ]]; then echo "FAIL: colmap binary not found/executable: $BIN"; exit 2; fi
if [[ ! -d "$DATASET/images" ]]; then echo "FAIL: dataset images dir not found: $DATASET/images"; exit 2; fi
if ! command -v python3 >/dev/null 2>&1; then echo "FAIL: python3 required"; exit 2; fi
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
DATASET="$(cd "$DATASET" && pwd)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Shared metric helpers (rss_gb, real_s, strip_ansi, model_reg, model_reproj).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_metrics_lib.sh"

echo "perf_baseline | bin=$BIN size=$SIZE mode=$([[ $UPDATE -eq 1 ]] && echo update || echo compare)"

mkdir -p "$WORK/images" "$WORK/sparse"
i=0; for f in $(ls "$DATASET/images" | sort); do
  [[ $i -ge $SIZE ]] && break; ln -sf "$DATASET/images/$f" "$WORK/images/$f"; i=$((i+1)); done

# --- Stage 1: feature extraction (CPU SIFT, first_octave=0, RAM budget) ---
/usr/bin/time -l "$BIN" feature_extractor --database_path "$WORK/db.db" \
  --image_path "$WORK/images" --FeatureExtraction.use_gpu 0 \
  --SiftExtraction.first_octave 0 --FeatureExtraction.max_memory_gb 4.0 \
  >"$WORK/feat.out" 2>"$WORK/feat.time" || { echo "FAIL: extraction"; exit 1; }
fe_t=$(real_s "$WORK/feat.time"); fe_g=$(rss_gb "$WORK/feat.time")

# --- Stage 2: exhaustive matching (CPU) ---
/usr/bin/time -l "$BIN" exhaustive_matcher --database_path "$WORK/db.db" \
  --FeatureMatching.use_gpu 0 \
  >"$WORK/match.out" 2>"$WORK/match.time" || { echo "FAIL: matching"; exit 1; }
ma_t=$(real_s "$WORK/match.time"); ma_g=$(rss_gb "$WORK/match.time")

# --- Stage 3: incremental mapping ---
/usr/bin/time -l "$BIN" mapper --database_path "$WORK/db.db" \
  --image_path "$WORK/images" --output_path "$WORK/sparse" \
  >"$WORK/map.out" 2>"$WORK/map.time" || { echo "FAIL: mapping"; exit 1; }
strip_ansi "$WORK/map.time" > "$WORK/map.clean"
mp_t=$(real_s "$WORK/map.clean"); mp_g=$(rss_gb "$WORK/map.clean")

reg=0; reproj="inf"
if [[ -d "$WORK/sparse/0" ]]; then
  "$BIN" model_analyzer --path "$WORK/sparse/0" >"$WORK/an.out" 2>&1
  reg=$(model_reg "$WORK/an.out")
  reproj=$(model_reproj "$WORK/an.out")
  [[ -z "$reg" ]] && reg=0
fi

# Hand the measured metrics to python3 for JSON store/diff (robust numerics).
BASELINE="$BASELINE" UPDATE="$UPDATE" SIZE="$SIZE" \
TIME_TOL="$TIME_TOL" RSS_TOL="$RSS_TOL" \
FE_T="$fe_t" FE_G="$fe_g" MA_T="$ma_t" MA_G="$ma_g" MP_T="$mp_t" MP_G="$mp_g" \
REG="$reg" REPROJ="$reproj" \
python3 - <<'PY'
import json, os, sys

def num(x, d=0.0):
    try: return float(x)
    except (TypeError, ValueError): return d

cur = {
    "size": int(num(os.environ["SIZE"])),
    "extract_s": num(os.environ["FE_T"]), "extract_gb": num(os.environ["FE_G"]),
    "match_s":   num(os.environ["MA_T"]), "match_gb":   num(os.environ["MA_G"]),
    "map_s":     num(os.environ["MP_T"]), "map_gb":     num(os.environ["MP_G"]),
    "reg_images": int(num(os.environ["REG"])),
    "reproj_px": num(os.environ["REPROJ"], float("inf")),
}
path = os.environ["BASELINE"]
update = os.environ["UPDATE"] == "1"

def show(m):
    print(f"  extract {m['extract_s']:.1f}s / {m['extract_gb']:.2f}GB | "
          f"match {m['match_s']:.1f}s / {m['match_gb']:.2f}GB | "
          f"map {m['map_s']:.1f}s / {m['map_gb']:.2f}GB | "
          f"reg {m['reg_images']} | reproj {m['reproj_px']:.3f}px")

print("Measured:")
show(cur)

if update:
    with open(path, "w") as f:
        json.dump(cur, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Baseline written to {path}")
    sys.exit(0)

if not os.path.exists(path):
    print(f"FAIL: no baseline at {path}; run with --update first.")
    sys.exit(2)

with open(path) as f:
    base = json.load(f)
print("Baseline:")
show(base)

time_tol = num(os.environ["TIME_TOL"]) / 100.0
rss_tol = num(os.environ["RSS_TOL"]) / 100.0
fails = []

# Time + RSS: regression if current exceeds baseline by more than tolerance.
for key, tol, label in [
    ("extract_s", time_tol, "extract time"), ("extract_gb", rss_tol, "extract RSS"),
    ("match_s", time_tol, "match time"),     ("match_gb", rss_tol, "match RSS"),
    ("map_s", time_tol, "map time"),         ("map_gb", rss_tol, "map RSS"),
]:
    b, c = base.get(key, 0.0), cur[key]
    if b > 0 and c > b * (1.0 + tol):
        fails.append(f"{label} regressed {c:.2f} vs {b:.2f} (+{100*(c/b-1):.0f}%, tol {100*tol:.0f}%)")

# Quality: fewer registered images or a worse reprojection error is a regression.
if cur["reg_images"] < base.get("reg_images", 0):
    fails.append(f"registered images dropped {cur['reg_images']} < {base['reg_images']}")
if cur["reproj_px"] > base.get("reproj_px", float("inf")) * 1.10:
    fails.append(f"reprojection error worsened {cur['reproj_px']:.3f} > {base['reproj_px']:.3f}px")

if fails:
    print("RESULT: FAIL")
    for f in fails: print("  - " + f)
    sys.exit(1)
print("RESULT: PASS (within tolerance of baseline)")
PY
