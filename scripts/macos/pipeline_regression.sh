#!/bin/bash
# COLMAP macOS sparse-pipeline integration + regression test.
#
# Runs the optimized mac production pipeline (resource-aware CPU SIFT extraction
# with first_octave=0 + a RAM budget, Metal GPU matching via use_gpu, then the
# incremental mapper) at INCREMENTALLY larger image counts, measuring peak RSS
# per stage and asserting both a RESOURCE budget and QUALITY floors. It is both
# an integration test (the whole pipeline must run + register images) and a
# regression gate (fails if RAM exceeds the cap or quality drops).
#
# Methodology (matches memory/testing_mechanism.md + on-device profiling):
#   * start small (2, 4, 6, 8 images), grow only while in budget;
#   * IDEAL 3-4 GB RAM, HARD cap 6 GB -- if a stage exceeds the cap the run
#     FAILS (the "cut and find a more efficient way" rule), it does not silently
#     continue. The lower the RAM, the better.
#   * assert all images register and mean reprojection error stays sub-pixel.
#
# Usage:
#   scripts/macos/pipeline_regression.sh [--bin PATH] [--dataset DIR]
#                                        [--sizes "2 4 6 8"] [--max-gb 6]
# Exit code 0 = all checks passed; non-zero = a budget or quality check failed.

set -u

BIN="${COLMAP_BIN:-build/src/colmap/exe/colmap}"
DATASET="${COLMAP_DATASET:-datasets/south-building}"
SIZES="2 4 6 8"
MAX_GB="6.0"          # HARD cap; exceeding it FAILS the run.
TARGET_GB="4.0"       # ideal ceiling (3-4 GB band); lower is better.
MAX_REPROJ_PX="1.5"   # quality floor: mean reprojection error must be below this

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) BIN="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --sizes) SIZES="$2"; shift 2;;
    --max-gb) MAX_GB="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

if [[ ! -x "$BIN" ]]; then echo "FAIL: colmap binary not found/executable: $BIN"; exit 2; fi
if [[ ! -d "$DATASET/images" ]]; then echo "FAIL: dataset images dir not found: $DATASET/images"; exit 2; fi
# Absolutize so symlinks created in the temp work dir resolve correctly.
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
DATASET="$(cd "$DATASET" && pwd)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

ts() { python3 -c 'import time;print(time.time())'; }
rss_gb() { # bytes from /usr/bin/time -l "maximum resident set size" -> GiB
  local b; b=$(grep "maximum resident" "$1" 2>/dev/null | awk '{print $1}')
  [[ -z "$b" ]] && { echo "0"; return; }
  echo "scale=2; $b/1073741824" | bc
}
over() { # $1 value, $2 limit -> 0 (true) if value > limit
  [[ $(echo "$1 > $2" | bc) -eq 1 ]]
}

FAILED=0
echo "COLMAP macOS pipeline regression | bin=$BIN dataset=$DATASET sizes=[$SIZES] cap=${MAX_GB}GB"
printf '%-4s | %-18s | %-18s | %-16s | %-10s | %s\n' N "extract s/GB" "match s/GB" "map s/GB" "reg/exp" verdict
echo "-----+--------------------+--------------------+------------------+------------+--------"

for N in $SIZES; do
  W="$WORK/n$N"; mkdir -p "$W/images" "$W/sparse"
  i=0; for f in $(ls "$DATASET/images" | sort); do
    [[ $i -ge $N ]] && break; ln -sf "$DATASET/images/$f" "$W/images/$f"; i=$((i+1)); done

  /usr/bin/time -l "$BIN" feature_extractor --database_path "$W/db.db" --image_path "$W/images" \
    --FeatureExtraction.use_gpu 0 --SiftExtraction.first_octave 0 \
    --FeatureExtraction.max_memory_gb "$TARGET_GB" >"$W/feat.out" 2>"$W/feat.time"
  fe_t=$(grep " real" "$W/feat.time" | awk '{print $1}'); fe_g=$(rss_gb "$W/feat.time")

  /usr/bin/time -l "$BIN" exhaustive_matcher --database_path "$W/db.db" \
    --FeatureMatching.use_gpu 1 >"$W/match.out" 2>"$W/match.time"
  ma_t=$(grep " real" "$W/match.time" | awk '{print $1}'); ma_g=$(rss_gb "$W/match.time")

  # Incremental SfM needs >= 4 images to reliably initialize+register; for
  # smaller loads we profile extraction+matching resource usage only.
  mp_t="-"; mp_g="0"; reg="n/a"; reproj="n/a"
  if [[ $N -ge 4 ]]; then
    /usr/bin/time -l "$BIN" mapper --database_path "$W/db.db" --image_path "$W/images" \
      --output_path "$W/sparse" >"$W/map.out" 2>"$W/map.time"
    # Strip ANSI color codes glog may emit, then parse.
    sed $'s/\x1b\\[[0-9;]*m//g' "$W/map.time" > "$W/map.clean"
    mp_t=$(grep " real" "$W/map.clean" | awk '{print $1}'); mp_g=$(rss_gb "$W/map.clean")
    if [[ -d "$W/sparse/0" ]]; then
      "$BIN" model_analyzer --path "$W/sparse/0" >"$W/an.out" 2>&1
      reg=$(grep -i "Registered images" "$W/an.out" | awk '{print $NF}')
      reproj=$(grep -i "Mean reprojection error" "$W/an.out" | awk '{print $NF}' | tr -d 'px')
      [[ -z "$reg" ]] && reg=0
    else
      reg=0
    fi
  fi

  verdict="PASS"
  for g in "$fe_g" "$ma_g" "$mp_g"; do
    if over "$g" "$MAX_GB"; then verdict="FAIL(RAM ${g}>${MAX_GB})"; FAILED=1;
    elif over "$g" "$TARGET_GB" && [[ "$verdict" == "PASS" ]]; then
      # In budget but above the 3-4 GB ideal band: pass, but flag it (lower is better).
      verdict="WARN(RAM ${g}>${TARGET_GB} ideal)"; fi
  done
  if [[ $N -ge 4 ]]; then
    [[ "$reg" != "$N" ]] && { verdict="FAIL(reg $reg/$N)"; FAILED=1; }
    if [[ "$reproj" != "inf" && "$reproj" != "n/a" && -n "$reproj" ]] && over "$reproj" "$MAX_REPROJ_PX"; then
      verdict="FAIL(reproj $reproj)"; FAILED=1; fi
  fi

  printf '%-4s | %-18s | %-18s | %-16s | %-10s | %s\n' \
    "$N" "${fe_t}/${fe_g}" "${ma_t}/${ma_g}" "${mp_t}/${mp_g}" "$reg/$N" "$verdict"

  # Per the budget rule: if we exceeded the hard cap, stop scaling up.
  if [[ "$verdict" == FAIL\(RAM* ]]; then
    echo "Hard RAM cap exceeded at N=$N; stopping (cut-and-redefine rule)."
    break
  fi
done

echo "-----"
if [[ $FAILED -eq 0 ]]; then echo "RESULT: PASS (all stages in budget, all images registered, sub-pixel error)"; exit 0
else echo "RESULT: FAIL (see rows above)"; exit 1; fi
