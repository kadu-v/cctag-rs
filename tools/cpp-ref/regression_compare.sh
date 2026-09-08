#!/usr/bin/env bash
# Independent oracle: upstream `regression --gen-ref` vs Rust FileLog XML via `regression --compare`.
# Also runs upstream against itself (two independent --gen-ref runs) to show the
# upstream's own run-to-run agreement under the same strict criterion.
# Usage: tools/cpp-ref/regression_compare.sh [image.png ...]  (default: samples + testdata/synth)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
REG="$ROOT/build/upstream/Darwin-arm64/regression"
[ -x "$REG" ] || { echo "missing $REG — run tools/cpp-ref/build.sh upstream"; exit 1; }
unset CCTAG_PARAMETERS_OVERRIDE
WORK="${WORK:-$ROOT/target/regression}"
rm -rf "$WORK"; mkdir -p "$WORK/src" "$WORK/ref" "$WORK/ref2" "$WORK/rust"
if [ $# -eq 0 ]; then set -- "$ROOT/CCTag/sample/01.png" "$ROOT/CCTag/sample/02.png" "$ROOT"/testdata/synth/*.png; fi
for img in "$@"; do cp "$img" "$WORK/src/"; done
( cd "$ROOT" && cargo build --release --features png --example detect >/dev/null 2>&1 )
gen_ref() { # <dst-dir>: one regression process per image (TestRunner wipes its dst dir; RNG state carries across images)
  local dst="$1"
  for img in "$WORK"/src/*.png; do
    local d; d="$(mktemp -d)"; cp "$img" "$d/"
    local name; name="$(basename "${img%.*}")"
    local out; out="$(mktemp -d)"
    ( cd "$WORK" && "$REG" --gen-ref --src-dir "$d" --dst-dir "$out" --parameters "$ROOT/testdata/defaultParameters.xml" >/dev/null 2>&1 )
    sed "s|<filename>.*</filename>|<filename>$WORK/src/$name.png</filename>|" "$out/$name.xml" > "$dst/$name.xml"
    rm -rf "$d" "$out"
  done
}
gen_ref "$WORK/ref"
gen_ref "$WORK/ref2"
for img in "$WORK"/src/*.png; do
  name="$(basename "${img%.*}")"
  "$ROOT/target/release/examples/detect" "$img" --filelog "$WORK/rust/$name.xml" >/dev/null
done
echo "== upstream vs upstream (regression --compare, epsilon 0.5)"
( cd "$WORK" && "$REG" --compare --src-dir "$WORK/ref" --dst-dir "$WORK/ref2" --epsilon 0.5 2>&1 | grep -v "^Processing\|CUDA override" ) || true
echo "== upstream vs rust (regression --compare, epsilon 0.5)"
( cd "$WORK" && "$REG" --compare --src-dir "$WORK/ref" --dst-dir "$WORK/rust" --epsilon 0.5 2>&1 | grep -v "^Processing\|CUDA override" ) || true
