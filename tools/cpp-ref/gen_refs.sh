#!/usr/bin/env bash
# Generate parity reference dumps with the ref-exact driver, one image per process.
# Usage: tools/cpp-ref/gen_refs.sh [image.png ...]   (default: CCTag/sample/01.png 02.png + testdata/synth/*.png)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DRIVER="$ROOT/build/ref-exact/driver/cctag_ref"
OUT="$ROOT/testdata/ref"
[ -x "$DRIVER" ] || { echo "missing $DRIVER — run tools/cpp-ref/build.sh ref-exact"; exit 1; }
unset CCTAG_PARAMETERS_OVERRIDE
if [ $# -eq 0 ]; then
  set -- "$ROOT/CCTag/sample/01.png" "$ROOT/CCTag/sample/02.png" "$ROOT"/testdata/synth/*.png
fi
for img in "$@"; do
  [ -f "$img" ] || continue
  name="$(basename "${img%.*}")"
  mkdir -p "$OUT/$name"
  echo "== $name"
  ( cd "$OUT/$name" && "$DRIVER" --image "$img" --mode e2e --warmup 0 --iters 1 --threads 1 --dump-dir "$OUT/$name" --json "$OUT/$name/markers_e2e.json" > "$OUT/$name/driver.log" 2>&1 )
  tail -n 3 "$OUT/$name/driver.log"
done
