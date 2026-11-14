# Lisa for macOS, iPhone and iPad

Current validation and the simulator bf16 compiler blocker are recorded in
[the shared Apple app architecture](../../specs/desktop.md#shared-apple-app).

This Xcode project shares its SwiftUI views, application state and Rust engine
between Apple Silicon macOS 15+ and iOS/iPadOS 18+. The existing AppKit executable
and `crates/lisa-ui/bundle.sh` remain available during migration.

Open `Lisa.xcodeproj`, select the **Lisa** scheme and choose a Mac, iPhone or iPad
destination. Select your signing team to install on a physical iOS device.
Install the Rust targets once:

```sh
rustup target add aarch64-apple-darwin aarch64-apple-ios aarch64-apple-ios-sim
```

Xcode's build phase builds `lisa-ui --lib` for the destination and links its
static archive. It copies the matching authenticated `kernels.bin`, source
notices and licenses into the app. No Rust or Metal shader source is downloaded
at runtime. Inference stays in process on the existing serial engine thread.
The Rust model implementations and arithmetic are shared with the CLI.

An unsigned simulator build:

```sh
xcodebuild -project apps/Lisa/Lisa.xcodeproj -scheme Lisa \
  -configuration Debug -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath target/apple-simulator CODE_SIGNING_ALLOWED=NO build
```

## Simulator integration check

Boot a simulator, then use its UUID from `xcrun simctl list devices`. The opt-in
Debug-only check downloads the curated Qwen3.5 0.8B 4-bit checkpoint through the
normal app downloader. It loads the model, generates replies locally, verifies
multi-turn recall, cancels generation and reopens a saved chat. It uses the real
AppStore and bridge; there are no mock replies. It creates test conversations
and retains them for inspection.

Read `sysctl hw.memsize` and `memory_pressure` first. Run inference and GPU tests
sequentially. The example cap must be lower than physical RAM minus 8 GiB.

```sh
xcrun simctl install "$SIMULATOR_ID" \
  target/apple-simulator/Build/Products/Debug-iphonesimulator/Lisa.app
SIMCTL_CHILD_LISA_RAM_CAP_GB=4 xcrun simctl launch --console-pty \
  "$SIMULATOR_ID" org.lisa.app --lisa-simulator-check
```

Inspect `Documents/simulator-check.json` inside the data directory returned by:

```sh
xcrun simctl get_app_container "$SIMULATOR_ID" org.lisa.app data
```

A passed simulator check does not certify physical-device GPU support, speed,
thermal behavior, app memory limits, background transitions or microphone access.
For an independent float32/bf16 compiler diagnostic, launch a Debug simulator
build with `--lisa-metal-probe` and inspect `Documents/metal-probe.json`.

The simulator uses host RAM and does not supply an iOS per-app memory limit.
Real devices additionally query `os_proc_available_memory()` before model load.
The standing physical RAM minus 8 GiB reserve is retained: low-memory devices
may run the interface but cannot admit a local model under that policy.

## Shared behavior

The app provides streaming chat, persisted conversations, curated model
downloads, image attachments and generation, a WebKit browser, on-device speech
input, spoken replies and the local API server. Platform wrappers are limited
to WebKit hosting, image display, audio sessions and app lifecycle.

Chats retain the existing JSON format under Application Support/Lisa/chats.
The Mac app shares the existing unsandboxed chat/model locations. iOS stores
files in its own application container. This does not synchronize data between
devices. The app requests cancellation and releases GPU state when iOS becomes
inactive; entering the background also stops its local server. Model reload and
saved-history replay happen on return to the foreground.

The UI currently renders inline Markdown and fenced code. The AppKit renderer's
custom tables, mathematical typesetting and slash-command picker have not been
ported. Voice, browser interstitials, image generation and API serving require
separate device-level acceptance checks before claiming complete desktop parity.
