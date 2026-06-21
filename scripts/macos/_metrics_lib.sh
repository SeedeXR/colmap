# Shared metric-extraction helpers for the macOS perf scripts
# (pipeline_regression.sh, perf_baseline.sh). Source this file; do not run it.
# Requires python3 (both callers already depend on it).

# Peak RSS in GiB from a `/usr/bin/time -l` stderr capture file ($1).
rss_gb() {
  local b
  b=$(grep "maximum resident" "$1" 2>/dev/null | awk '{print $1}')
  [[ -z "$b" ]] && { echo "0"; return; }
  python3 -c "print(round($b/1073741824, 3))"
}

# Wall-clock seconds (the "real" line) from a `/usr/bin/time -l` capture ($1).
real_s() { grep " real" "$1" 2>/dev/null | tail -1 | awk '{print $1}'; }

# Strip glog ANSI color codes from file $1 to stdout.
strip_ansi() { sed $'s/\x1b\\[[0-9;]*m//g' "$1"; }

# Registered-image count from a `colmap model_analyzer` capture file ($1).
model_reg() { grep -i "Registered images" "$1" 2>/dev/null | awk '{print $NF}'; }

# Mean reprojection error (px, unit stripped) from a model_analyzer capture ($1).
model_reproj() {
  grep -i "Mean reprojection error" "$1" 2>/dev/null | awk '{print $NF}' | tr -d 'px'
}
