#!/bin/sh
# Xcode invokes this for the active Apple destination. Kernels match the archive.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
case "${PLATFORM_NAME:-macosx}" in
    macosx) RUST_TARGET=aarch64-apple-darwin ;;
    iphoneos) RUST_TARGET=aarch64-apple-ios ;;
    iphonesimulator) RUST_TARGET=aarch64-apple-ios-sim ;;
    *) printf 'Unsupported platform: %s\n' "$PLATFORM_NAME" >&2; exit 1 ;;
esac
case " ${ARCHS:-arm64} " in
    *" x86_64 "*) printf 'Lisa requires an Apple Silicon build host and arm64 destination.\n' >&2; exit 1 ;;
esac
CARGO=${CARGO:-"$HOME/.cargo/bin/cargo"}
PROFILE=dist
if [ "${CONFIGURATION:-Release}" = Debug ]; then PROFILE=dev; fi
# A separate directory avoids locking the CLI build while Xcode links the app.
CARGO_TARGET_DIR="$ROOT/target/apple-rust"
export CARGO_TARGET_DIR
cd "$ROOT"
"$CARGO" build --locked -p lisa-app --lib --target "$RUST_TARGET" --profile "$PROFILE"
OUTPUT=dist
if [ "$PROFILE" = dev ]; then OUTPUT=debug; fi
LINK_DIR="$ROOT/apps/Lisa/build/${PLATFORM_NAME:-macosx}/${CONFIGURATION:-Release}"
mkdir -p "$LINK_DIR"
ln -sf "$CARGO_TARGET_DIR/$RUST_TARGET/$OUTPUT/liblisa_app.a" "$LINK_DIR/liblisa_app.a"
RESOURCES="$TARGET_BUILD_DIR/$UNLOCALIZED_RESOURCES_FOLDER_PATH"
mkdir -p "$RESOURCES/Licenses"
cp "$CARGO_TARGET_DIR/$RUST_TARGET/$OUTPUT/kernels.bin" "$RESOURCES/kernels.bin"
cp LICENSE NOTICE "$RESOURCES/"
cp crates/lisa-metal/src/shaders/deepseek41/LICENSE "$RESOURCES/Licenses/DS4.txt"
cp crates/lisa-gguf/src/LICENSE-GGML "$RESOURCES/Licenses/GGML.txt"
cp crates/lisa-models/src/flux/LICENSE.flux3 "$RESOURCES/Licenses/Apache-2.0.txt"
if [ "${PLATFORM_NAME:-macosx}" = macosx ]; then
    cp apps/Lisa/assets/Lisa.icns "$RESOURCES/Lisa.icns"
fi
