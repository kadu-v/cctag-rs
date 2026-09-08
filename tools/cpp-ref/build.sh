#!/usr/bin/env bash
# Build the C++ CCTag reference in up to three configurations (static libs) and
# the headless driver (tools/cpp-ref/driver) against each:
#   upstream    : pristine tree, TBB parallel        -> speed baseline, Tier C oracle
#   perf-serial : -DCCTAG_SERIALIZE=ON               -> speed baseline for Rust Parity mode
#   ref-exact   : perf-serial + -ffp-contract=off + parity patches + CCTAG_PARITY_DUMP
# Usage: tools/cpp-ref/build.sh [upstream|perf-serial|ref-exact|all] [--skip-apps]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SRC="$ROOT/CCTag"
BUILD="$ROOT/build"
EIGEN_SRC="$HERE/third_party/eigen-3.4.0"
EIGEN_PREFIX="$HERE/third_party/eigen-install"
WHAT="${1:-all}"
SKIP_APPS="${2:-}"
NCPU="$(sysctl -n hw.ncpu)"

unset CCTAG_PARAMETERS_OVERRIDE

if [ ! -f "$EIGEN_PREFIX/share/eigen3/cmake/Eigen3Config.cmake" ]; then
  echo "== installing Eigen 3.4.0 headers to $EIGEN_PREFIX"
  cmake -S "$EIGEN_SRC" -B "$HERE/third_party/eigen-build" -DCMAKE_INSTALL_PREFIX="$EIGEN_PREFIX" \
        -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF -DEIGEN_BUILD_PKGCONFIG=OFF >/dev/null
  cmake --install "$HERE/third_party/eigen-build" >/dev/null
fi

APPS=ON; [ "$SKIP_APPS" = "--skip-apps" ] && APPS=OFF

common=(
  -DCMAKE_BUILD_TYPE=Release
  -DCCTAG_WITH_CUDA=OFF
  -DCCTAG_BUILD_APPS=$APPS
  -DCCTAG_BUILD_TESTS=ON
  -DCCTAG_BUILD_DOC=OFF
  -DBUILD_SHARED_LIBS=OFF
  -DCMAKE_PREFIX_PATH=/opt/homebrew
  -DEigen3_DIR="$EIGEN_PREFIX/share/eigen3/cmake"
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
)

build_one() {
  local name="$1"; shift
  local dir="$BUILD/$name"
  echo "== configuring $name"
  cmake -S "$SRC" -B "$dir" "${common[@]}" "$@" 2>&1 | grep -E "error|Error" | head -n 10 || true
  echo "== building $name"
  cmake --build "$dir" -j"$NCPU" 2>&1 | grep -E "error|Built target" | tail -n 20
  echo "== installing $name -> $dir/install"
  cmake --install "$dir" --prefix "$dir/install" >/dev/null
  echo "== building driver for $name"
  cmake -S "$HERE/driver" -B "$dir/driver" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$dir/install;/opt/homebrew" \
        -DEigen3_DIR="$EIGEN_PREFIX/share/eigen3/cmake" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 2>&1 | grep -E "error|Error" | head -n 10 || true
  cmake --build "$dir/driver" -j"$NCPU" 2>&1 | grep -E "error|Built target" | tail -n 5
  echo "== done: $dir (driver: $dir/driver/cctag_ref)"
}

# ref-exact needs the parity patches applied on a local branch of CCTag/.
apply_patches() {
  local cur; cur="$(git -C "$SRC" rev-parse --abbrev-ref HEAD)"
  if [ "$cur" != "parity-harness" ]; then
    if git -C "$SRC" rev-parse --verify -q parity-harness >/dev/null; then
      git -C "$SRC" checkout -q parity-harness
    else
      echo "== creating branch parity-harness in CCTag/ and applying patches"
      git -C "$SRC" checkout -q -b parity-harness
      for p in "$HERE"/patches/*.patch; do
        echo "   applying $(basename "$p")"; git -C "$SRC" am -q "$p"
      done
    fi
  fi
}
restore_upstream() {
  local cur; cur="$(git -C "$SRC" rev-parse --abbrev-ref HEAD)"
  if [ "$cur" = "parity-harness" ]; then git -C "$SRC" checkout -q develop; fi
}

case "$WHAT" in
  upstream)    restore_upstream; build_one upstream ;;
  perf-serial) restore_upstream; build_one perf-serial -DCCTAG_SERIALIZE=ON ;;
  ref-exact)   apply_patches;    build_one ref-exact -DCCTAG_SERIALIZE=ON -DCCTAG_PARITY_DUMP=ON \
                                   -DCMAKE_CXX_FLAGS="-ffp-contract=off" ;;
  all)         restore_upstream; build_one upstream; build_one perf-serial -DCCTAG_SERIALIZE=ON
               apply_patches;    build_one ref-exact -DCCTAG_SERIALIZE=ON -DCCTAG_PARITY_DUMP=ON \
                                   -DCMAKE_CXX_FLAGS="-ffp-contract=off" ;;
  *) echo "unknown target $WHAT"; exit 2 ;;
esac
