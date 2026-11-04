#!/bin/sh
# Build a local macOS application bundle, including its Finder icon.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
cargo build --profile dist -p lisa-ui
TARGET=${CARGO_TARGET_DIR:-"$ROOT/target"}
APP="$TARGET/dist/Lisa.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$TARGET/dist/lisa-ui" "$APP/Contents/MacOS/lisa-ui"
cp crates/lisa-ui/Info.plist "$APP/Contents/Info.plist"
cp crates/lisa-ui/assets/Lisa.icns "$APP/Contents/Resources/Lisa.icns"
cp "$TARGET/dist/kernels.bin" "$APP/Contents/Resources/kernels.bin"
cp LICENSE NOTICE "$APP/Contents/Resources/"
mkdir -p "$APP/Contents/Resources/Licenses"
cp crates/lisa-mlx/src/shaders/deepseek41/LICENSE "$APP/Contents/Resources/Licenses/DS4.txt"
cp crates/lisa-gguf/src/LICENSE-GGML "$APP/Contents/Resources/Licenses/GGML.txt"
cp crates/lisa-engine/src/models/flux3/LICENSE "$APP/Contents/Resources/Licenses/Apache-2.0.txt"
# Cargo's packed debuginfo creates the matching lisa-ui.dSYM beside the binary.
test -d "$TARGET/dist/lisa-ui.dSYM"
strip "$APP/Contents/MacOS/lisa-ui"
codesign --force --sign - "$APP"
bytes=$(find "$APP" -type f -exec stat -f '%z' {} + | awk '{total += $1} END {printf "%.0f", total}')
if [ "$bytes" -gt 10000000 ]; then
    printf 'App bundle exceeds the 10 MB limit: %s bytes\n' "$bytes" >&2
    exit 1
fi
printf 'App bundle: %s bytes; debug symbols: %s/dist/lisa-ui.dSYM\n' "$bytes" "$TARGET"
printf '%s\n' "$APP"
