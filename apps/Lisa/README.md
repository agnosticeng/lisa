# Lisa for macOS, iPhone and iPad

Current validation and the simulator bf16 compiler blocker are recorded in
[the Apple app spec](../../specs/apple-app.md).

This Xcode project shares its SwiftUI views, application state and Rust engine
between Apple Silicon macOS 15+ and iOS/iPadOS 18+. This is the sole app frontend.
`crates/lisa-app` contains its Rust services and C bridge; `sh apps/Lisa/bundle.sh`
builds the locally signed macOS distribution.

Open `Lisa.xcodeproj`, select the **Lisa** scheme and choose a Mac, iPhone or iPad
destination. Select your signing team to install on a physical iOS device.
Choose a unique bundle identifier in Signing & Capabilities if your team cannot
register the default `org.lisa.app`. Team and identifier can also be supplied as
`DEVELOPMENT_TEAM=... PRODUCT_BUNDLE_IDENTIFIER=...` Xcode build overrides.
Install the Rust targets once:

```sh
rustup target add aarch64-apple-darwin aarch64-apple-ios aarch64-apple-ios-sim
```

Xcode's build phase builds `lisa-app --lib` for the destination and links its
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

## Background generation

The iOS 26+ integration uses `BGContinuedProcessingTask` with the GPU resource,
real generation progress, expiration handling and completion after chat storage.
The system and device must grant background GPU access. App-switcher force-quit
terminates tasks. WebKit pages needing consent still require returning to Lisa.

The default build retains personal-team signing and foreground generation.
Apple rejects the Background GPU Access capability for personal development teams.
With an eligible team and a profile supporting that capability, set
`LISA_IOS_ENTITLEMENTS=Lisa-iOS.entitlements` when building for a physical device.
The entitlement is scoped to iPhoneOS, not macOS or Simulator. Unsupported devices,
older iOS versions or rejected task requests fall back to foreground generation;
the screen stays awake until generation completes or is cancelled.

For a real background image check, add `--lisa-image-check --lisa-background-check`
to the integration arguments with a downloaded image model. Once the task is
authorized, leave Lisa using Home or lock the phone. The report must show both
background GPU authorization and an actual background scene transition followed
by a saved PNG; direct lifecycle calls alone do not validate OS permission.

## Integration check

The opt-in Debug-only check runs on macOS, iOS devices and Simulator. It downloads
the curated Qwen3.5 0.8B 4-bit checkpoint through the
normal app downloader. Add `--lisa-check-model LiquidAI/LFM3-1.7B-MLX-4bit`
to exercise LFM3 through the same checks. It loads the model, generates replies locally, verifies
multi-turn recall, cancels generation and reopens a saved chat. It uses the real
AppStore and bridge; there are no mock replies. It creates test conversations
and retains them for inspection. On iOS it also invokes the scene lifecycle and
memory-warning handlers to check unload/reload and history replay. These direct
handler checks do not certify actual OS suspension or memory-pressure delivery.

Add `--lisa-integration-check` to the scheme's Run arguments for any destination,
or launch a Debug Mac build's `Lisa.app/Contents/MacOS/Lisa` executable with that
argument. Inspect `Documents/integration-check.json` in the app's container
(the user's Documents directory on macOS). For a simulator, boot it and use its
UUID from `xcrun simctl list devices`:

Read `sysctl hw.memsize` and `memory_pressure` first. Run inference and GPU tests
sequentially. The example cap must be lower than physical RAM minus 8 GiB.

```sh
xcrun simctl install "$SIMULATOR_ID" \
  target/apple-simulator/Build/Products/Debug-iphonesimulator/Lisa.app
xcrun simctl launch --console-pty \
  "$SIMULATOR_ID" org.lisa.app --lisa-integration-check --ram-cap-gb 4
```

Inspect `Documents/integration-check.json` inside the data directory returned by:

```sh
xcrun simctl get_app_container "$SIMULATOR_ID" org.lisa.app data
```

A passed simulator check does not certify physical-device GPU support, speed,
thermal behavior, app memory limits, background transitions or microphone access.
The `--lisa-simulator-check` argument writes `simulator-check.json`.
For an independent float32/bf16 compiler diagnostic, launch any Debug
build with `--lisa-metal-probe` and inspect `Documents/metal-probe.json`.

The simulator uses host RAM and does not supply an iOS per-app memory limit.
Real devices query `os_proc_available_memory()` on every admission and retain
512 MiB of headroom from the smaller of that allowance and physical RAM. A zero
allowance refuses loading. The `--ram-cap-gb` launch argument can only lower
the result. This
permits small models on 8 GiB iPhones while keeping the Mac's 8 GiB reserve.
Model size is an initial admission estimate; long contexts and image workloads
can still exhaust memory. On a memory warning, Lisa cancels work and unloads the
model without automatically loading it again.

## Shared behavior

On iPhone, the sidebar button or a right swipe opens a flat left-side panel with chat
search, models, settings and activity. The rounded conversation pane moves right;
tap it or swipe left on it to close. Selecting a conversation also closes the
panel; swipe left on a chat row to reveal Delete. iPad and Mac retain their split
sidebar. Empty chats keep the conversation area clear; activity logs live in the
Activity view, not below the composer. Models exceeding the current device budget
are hidden; downloaded models appear first among those that fit. Each conversation keeps
its selected model and reloads it when reopened, including after app relaunch.
New installations default to Qwen3.5 0.8B MLX 4-bit and download/load it on launch
when it fits. A saved conversation's pinned model takes precedence.

The app provides streaming chat, persisted conversations, curated model
downloads, image attachments and generation, a WebKit browser, on-device speech
input, spoken replies and the local API server. Platform wrappers are limited
to WebKit hosting, image display, audio sessions and app lifecycle.

Chats use the shared JSON format in the app data directory. The Mac app uses
unsandboxed chat/model locations; iOS stores models in its application container,
excluded from iCloud backups. The application configures that cache
explicitly before workers start; desktop environment defaults are not writable
model locations on iOS. Download failures are shown in the UI and device console.
This does not synchronize data between
devices. Without an authorized background generation task, entering the iOS
background requests cancellation and releases GPU state and the local server.
Model reload and saved-history replay happen on return to the foreground.
Transient inactivity alone does not release the model.

Custom mathematical typesetting and a slash-command picker are not
implemented in the SwiftUI interface. Voice, browser interstitials, image generation and API serving require
separate device-level acceptance checks before claiming complete desktop parity.

The SwiftUI chat renders Markdown block structure with Foundation's full parser,
including headings, nested lists, quotes, fenced code and tables. Run its focused
regression check without loading a model:

```sh
swiftc apps/Lisa/Sources/ChatLayout.swift apps/Lisa/Sources/MarkdownMessage.swift apps/Lisa/Tests/MarkdownCheck.swift -o /tmp/lisa-markdown-check
/tmp/lisa-markdown-check
```

## Web search checks and limits

The browser stays hidden during searches and page reads. It appears only when
Google requires consent or verification, and hides again after extraction.
Closing that prompt cancels the pending browser request. Lisa never chooses a
consent option or completes verification automatically.

The web agent opens only links from the latest Google results. It selects a
result number; full mobile redirect URLs remain in the navigator. Direct search
requests search Google and open the first result before inference. Failed
retrieval cannot fall back to an answer from model memory. Web replies show source links and compact extracts from the pages actually read.
Release listings render as film titles with source dates/genres; ratings, buttons
and navigation are omitted. Other pages use short quoted excerpts. Model-written summaries are not used for those replies, because small
models can invent titles even after reading sources.
Excerpts preserve the source wording; they do not certify that a page is current
or that its claims are true.

Every chat receives the current UTC date in its runtime system prompt. Saved
instructions and visible user messages are unchanged. Reopening a chat or sending
a message on a new date refreshes that context and invalidates stale session state.

On physical iOS devices, web inference limits prompts to 2,048 tokens and each
pass to 512 output tokens. Older conversation turns are removed from the temporary
web context before current evidence is shortened; saved history is retained.
A scoped Metal allowance permits at most 512 MiB of additional native allocations,
or the remaining app budget if smaller. Pending work drains outside allocator
locks before completed, unleased buffers are reclaimed. An allocation that still
cannot fit returns an error. Mac/Simulator scheduling remains unchanged.

Run the extraction regression from the repository root (no model required):

```sh
swiftc apps/Lisa/Tests/BrowserScriptCheck.swift -parse-as-library -o /tmp/lisa-browser-script-check
/tmp/lisa-browser-script-check
```

For an actual browser/Metal integration check, launch a Debug app with
`--lisa-web-check`. Optional `--lisa-web-query "..."` supplies the request;
`--lisa-web-replay <chat-id>` copies completed turns from an existing chat into a
new test conversation. The original conversation is not edited. Inspect
`Documents/web-check.json` for search extraction, page reading, completed reply,
hidden browser and absence of iOS memory warnings. The report records the test
chat ID, diagnostic logs and reply; passing it does not certify factual accuracy
or a live consent challenge.
