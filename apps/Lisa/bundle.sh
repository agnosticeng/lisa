#!/bin/sh
# Build the shared SwiftUI app for local macOS distribution.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
TARGET=${CARGO_TARGET_DIR:-"$ROOT/target"}
mkdir -p "$TARGET"
TARGET=$(CDPATH= cd -- "$TARGET" && pwd)
DERIVED="$TARGET/apple-dist"
xcodebuild -project "$ROOT/apps/Lisa/Lisa.xcodeproj" -scheme Lisa \
    -configuration Release -destination 'generic/platform=macOS' \
    -derivedDataPath "$DERIVED" CODE_SIGNING_ALLOWED=NO build
PRODUCTS="$DERIVED/Build/Products/Release"
APP="$PRODUCTS/Lisa.app"
SYMBOLS="$PRODUCTS/Lisa.app.dSYM"
test -d "$SYMBOLS"
test -f "$APP/Contents/Resources/kernels.bin"
codesign --force --sign - "$APP"
bytes=$(find "$APP" -type f -exec stat -f '%z' {} + | awk '{total += $1} END {printf "%.0f", total}')
mkdir -p "$TARGET/dist"
# Replace complete products so files from the retired AppKit bundle cannot linger.
STAGING=$(mktemp -d "$TARGET/dist/.lisa-bundle.XXXXXX")
trap 'rm -rf "$STAGING"' EXIT HUP INT TERM
ditto "$APP" "$STAGING/Lisa.app"
ditto "$SYMBOLS" "$STAGING/Lisa.app.dSYM"
rm -rf "$TARGET/dist/Lisa.app" "$TARGET/dist/Lisa.app.dSYM"
mv "$STAGING/Lisa.app" "$TARGET/dist/Lisa.app"
mv "$STAGING/Lisa.app.dSYM" "$TARGET/dist/Lisa.app.dSYM"
printf 'App bundle: %s bytes; debug symbols: %s/dist/Lisa.app.dSYM\n' "$bytes" "$TARGET"
printf '%s\n' "$TARGET/dist/Lisa.app"
