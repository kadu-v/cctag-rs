#!/usr/bin/env bash
# Run the unmodified upstream build on each reference image and store its output
# next to the ref-exact dumps (testdata/ref/<name>/markers_upstream.json).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DRIVER="$ROOT/build/upstream/driver/cctag_ref"
[ -x "$DRIVER" ] || { echo "missing $DRIVER — run tools/cpp-ref/build.sh upstream"; exit 1; }
unset CCTAG_PARAMETERS_OVERRIDE
for d in "$ROOT"/testdata/ref/*/; do
  name="$(basename "$d")"
  img=""
  for c in "$ROOT/CCTag/sample/$name.png" "$ROOT/testdata/synth/$name.png"; do [ -f "$c" ] && img="$c"; done
  [ -n "$img" ] || continue
  echo "== $name"
  "$DRIVER" --image "$img" --mode e2e --warmup 0 --iters 1 --json "$d/markers_upstream.json" > "$d/upstream.log" 2>&1
  tail -n 2 "$d/upstream.log"
done
