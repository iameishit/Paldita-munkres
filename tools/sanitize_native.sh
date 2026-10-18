#!/bin/sh
# Build the native kernel with AddressSanitizer + UndefinedBehaviorSanitizer and fuzz it.
#   sh tools/sanitize_native.sh [iterations] [seed]
set -eu
cd "$(dirname "$0")/.."
ITER="${1:-5000}"; SEED="${2:-0}"
PYINC="$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')"
SUFFIX="$(python3 -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')"
OUT="build/sanitize/munkres"
rm -rf build/sanitize && mkdir -p "$OUT"
cp -r src/munkres/*.py src/munkres/py.typed "$OUT"/ 2>/dev/null || cp -r src/munkres/*.py "$OUT"/
gcc -std=c99 -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined \
    -Wall -Wextra -Wpedantic -shared -fPIC -I"$PYINC" src/munkres/_native/lapmodule.c -o "$OUT/_lapc$SUFFIX"
mkdir -p "$OUT/_native"; cp src/munkres/_native/*.c "$OUT/_native/" 2>/dev/null || true
ASAN_LIB="$(gcc -print-file-name=libasan.so)"
LD_PRELOAD="$ASAN_LIB" ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
    PYTHONPATH="build/sanitize" python3 - <<PY
import munkres, sys
assert "build/sanitize" in munkres.__file__, munkres.__file__
sys.argv = ["fuzz", "--iterations", "$ITER", "--seed", "$SEED"]
exec(open("tools/fuzz_native.py").read().replace("sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / \"src\"))", "pass"))
PY
