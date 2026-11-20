# Apple app

The shared SwiftUI app lives in [apps/Lisa](../apps/Lisa/README.md); its Rust
worker, storage, downloads and C bridge live in [lisa-app](../crates/lisa-app/src/).
Open the Xcode project for macOS 15+ or iOS/iPadOS 18+, or build the local Mac
bundle with `sh apps/Lisa/bundle.sh`. Xcode and an Apple Silicon host are required.
The editable icon and generated desktop assets live in [assets](../apps/Lisa/assets/).

The bundle script produces `target/dist/Lisa.app` and reports its complete size,
including UI, engine, kernels, icon and notices. The retired AppKit bundle's
12 MB limit does not apply to the SwiftUI distribution. The `dist` profile uses
full LTO; inference/runtime optimization remains at level 3. The matching
`target/dist/Lisa.app.dSYM` stays outside the stripped app for symbolication.
Model weights are separate. Xcode bundles the Rust target's matching `kernels.bin`;
never copy a kernel package from another build.

## Navigation and saved chats

Both platforms show New chat, Models, Settings, Activity and Search, followed by
saved conversations. Search expands its input on demand. Mac and iPad use a split
sidebar; iPhone uses a sliding drawer opened by its sidebar icon or a right swipe.
Activity uses a terminal icon and holds diagnostic logs outside the composer.
iOS chat rows support swipe-to-delete; Mac rows expose deletion in their context menu.
Empty unsent chats do not clutter the history list.

[Chat storage](../crates/lisa-app/src/chats.rs) retains the existing JSON format,
model identity, drafts and owned attachments under Application Support/Lisa/chats.
The Mac uses its existing unsandboxed location; iOS uses its application container.
Reopening a conversation reloads its pinned model and replays history. GPU cache
state does not persist across restarts. New installations default to Qwen3.5 0.8B
MLX 4-bit when it fits. Downloaded models appear first; models exceeding the live
memory budget are hidden. Runtime admission remains authoritative.

SwiftUI owns image display, WebKit, speech and permissions. Generated images and
attachments use the shared Rust services. Voice, image generation, serving and
browser interstitials need separate device acceptance; do not infer those from
text-chat checks. The README image shows a real FLUX.2 Klein 4B generation in
the shared SwiftUI app, alongside saved Qwen 3.8 Flash-Next conversations.
Generated images use rounded, proportionally sized cards with preview, save,
share and (on Mac) Finder actions.

On iOS 26+, user-started generations request a continued-processing task with
background GPU access when supported. Authorized work retains its model across
background transitions and reports progress through the system Live Activity.
Completion saves the chat, releases a backgrounded model and finishes the task.
Expiration or memory pressure requests cancellation and ordered model release.
Force-quitting still terminates work; interrupted generation is not resumed.

The default personal-team build omits the optional GPU entitlement: Xcode rejects
Background GPU Access for personal development teams. This build remains
foreground-only and disables automatic screen locking during generation. Brief
inactive transitions do not unload the model; entering the background without
GPU authorization cancels work. See the [signing instructions](../apps/Lisa/README.md#background-generation)
for eligible-team builds. Background GPU execution has not been device-validated
with an entitled provisioning profile.

## Models and downloads

The allowlist is [catalog.rs](../crates/lisa-app/src/catalog.rs). Hub search does not add new
catalog entries. Locally cached checkpoints are shown only when their repository
is in the same allowlist.
Type icons distinguish image generators; download actions use an icon.

[Discovery](../crates/lisa-app/src/models.rs) shares checkpoint identities and
snapshot ordering with the engine. It validates available package/shard files
before listing an allowed local model and excludes decision-only Laya checkpoints
from chat. Filtering the list does not delete cached files. Local size accounting
follows Hub links, counts each inode once and avoids directory cycles. The
displayed RAM budget uses the engine's shared budget calculation.

Compatible MLX Qwen 3.8 27B targets use DSpine by default. A curated Qwen 27B
download first fetches the shared DSpine companion, then the target without
its optional MLX `mtp/` sidecar; one progress indicator covers both packages.
Completed companion downloads are reused across target quantizations. Existing
local targets check their companion before every load, reusing a complete
local copy without a network request. Missing or truncated companions start an
automatic download with progress in the model row; target loading then resumes.
Availability checks validate the config and safetensors payload bounds without
allocating GPU weights. A failed companion
download fails the operation instead of silently reverting to MTP. The UI
includes DSpine in package/RAM estimates and excludes unused MTP residency.
The BF16 companion adds roughly 3.8 GB over the curated MLX MTP sidecar; see
[Qwen notes](models/qwen.md#dspine-block-drafter). This path remains experimental.

Downloads use the official `hf-hub` 0.4 synchronous Rust client for authenticated
metadata, resumable HTTP transfers and cache writes. Its async and native Xet
transfer stacks are excluded to limit app size. The progress ring uses transfer callbacks
for the selected files, including cache hits, instead of counting unrelated
cached blobs. A `.lisa-incomplete` marker hides a snapshot from Lisa discovery
until all selected files pass size checks; retrying reuses completed files and
repairs truncated entries. `HF_HUB_CACHE`, `HF_HOME`, `HF_TOKEN`, token files and
`HF_ENDPOINT` are supported. If ambient credentials
are rejected, Lisa retries public metadata anonymously; private/gated access
still reports the authentication failure.

The list separates download size from resident memory estimates. DeepSeek's
Engram table is disk-only. GGUF packages select the required files and skip
legacy MLX preparation. Original model licenses/notices remain with downloads.
The catalog's memory check is an estimate; runtime admission still accounts for
caches and scratch. Only one model is loaded at a time.

Text chat and the app server use automatic MTP where the checkpoint supports
it. This enables GLM's embedded head; it does not create an absent DeepSeek
head or prove a speedup. Refer to [model status](README.md#model-status).

## Shared Apple app

[apps/Lisa](../apps/Lisa/README.md) is the single SwiftUI interface for macOS
and iOS/iPadOS. [lisa-app](../crates/lisa-app/src/) supplies the shared Rust
services as `liblisa_app.a`. The retired AppKit executable and its UI dependencies
have been removed.

`backend.rs` owns the model worker, session replay and streamed events for both
platforms. Models never cross the worker-thread boundary. `browser.rs` owns extraction
and browser messages; each platform supplies its WebKit presentation/lifecycle.
`bridge.rs` exposes owned UTF-8 JSON requests/responses through two C functions.
Swift calls on a serial queue and releases returned Rust strings explicitly.
SwiftUI owns navigation, view state and permissions, while chat persistence keeps
the existing Rust schema. Only owned attachment paths may be relocated after
an iOS application container changes.

On iOS, Swift supplies Application Support/Lisa and excludes its `models`
directory from backups. The bridge configures the engine's process-lifetime Hub
cache there before starting workers; downloads and discovery use the same root.
This avoids the unwritable desktop `~/.cache` default without mutating environment
variables in a multithreaded Swift process. Download errors reach the UI and
device console, with filesystem paths retained in diagnostic errors.

The Xcode build bundles matching kernels and licenses for its Rust target.
iOS requires version 18+ for compiler APIs. Physical devices use the smaller of
physical RAM and the current additional per-app allowance, minus 512 MiB;
`LISA_RAM_CAP_GB` can only lower it. The allowance is read on every admission,
using Apple's [os_proc_available_memory](https://developer.apple.com/documentation/os/os_proc_available_memory)
contract, and is not a guarantee against memory pressure during generation.
Simulator uses the Mac host physical-RAM-minus-8-GiB budget; it
does not certify a physical iPhone/iPad's capabilities. The current Simulator
BF16 compiler-service failure blocks inference acceptance; do not change weights
or arithmetic to hide it.

Cancellation occurs at generation callbacks, not inside an active forward.
An iOS background scene without authorized continued processing requests
cancellation/model release; returning reloads and replays saved history. Memory warnings release through the same worker and
disable automatic reload. Incomplete turns are not committed. Suspension, browser,
voice, image generation and serving need physical-device acceptance. SwiftUI
renders headings, nested lists, quotes, fenced code and tables. Custom mathematical
typesetting and a slash-command picker are not implemented.

### Apple validation — 2026-11-19

Xcode 27.2 Debug builds passed for Apple Silicon macOS, arm64 Simulator and a
signed physical iPhone. The shared app's real Qwen3.5 0.8B MLX 4-bit check passed
on macOS 27.0.1 with a 4 GiB cap and iPhone 15 Pro Max / iOS 27.2.1 with a 2 GiB
cap. Both exercised model load, streamed replies, multi-turn recall,
cancellation, persisted chats and replay excluding the cancelled turn. The phone
also passed direct inactive/active and memory-warning handler checks: model
release, reload/history replay and no automatic reload after memory pressure.
The phone downloaded the checkpoint through the normal app downloader into its
Application Support cache. These are functional checks, not performance or
strict cross-device numerical parity measurements.

The iOS 27.2 Simulator float32 copy probe passed; both default-language and
explicit-MSL-3.2 BF16 copies failed with `XPC_ERROR_CONNECTION_INTERRUPTED`.
Physical-device chat success does not remove this independent Simulator blocker.
Actual OS suspension, delivered low-memory warnings, long contexts, additional
devices, voice, browser, image generation and serving remain separate acceptance
work. The Debug check briefly disables auto-lock and restores its prior setting.

The compact iPhone UI was inspected in the iOS 27.2 simulator: sidebar-button and
horizontal-swipe drawer navigation, chat selection, swipe-left Delete, a clear
empty conversation, centered composer controls and model availability under a
2 GiB test cap. The final library hides models that exceed the live budget.
Model pins persist in the existing
chat JSON. Startup reloads the selected chat's model; a fresh install downloads
and loads Qwen3.5 0.8B when its admission check passes. Runtime model failures are
shown as alerts; diagnostic logs remain in Activity.

### Web search correction — 2026-11-19

The iPhone 15 Pro Max / iOS 27.2.1 reproduction used Qwen3.5 0.8B MLX
4-bit and the French movie-search request after two saved conversation turns.
The original 932-token web prefill ended with signal 9 on two runs. Delivered
memory warnings were also observed during an intermediate unrestricted run.
The bounded Metal path completed web prefills up to 2,034 tokens without a
memory warning or process termination. It drains work outside allocator locks
before reclaiming completed buffers, with a 512 MiB transient allowance. No
shader arithmetic, reduction order or forward shapes were changed. The focused
allocation test compares an unbounded 24-dispatch chain with two bounded runs,
bit for bit, and checks rejection while live storage remains owned. These checks
are functional/lifetime evidence, not a throughput claim; the failing unbounded
phone path cannot supply a completed interleaved performance baseline.

Google mobile cards use ARIA headings and opaque `/goto` redirects. Both are
now extracted, including cards whose offscreen `innerText` is empty. WebKit keeps
a nonzero viewport while hidden. Navigation is restricted to exact Google result
links; the model selects result numbers instead of constructing URLs. Google
consent/verification alone presents the browser. Consent is never automated.

Reading valid pages did not make the 0.8B checkpoint's summaries reliable: it
confused page titles with films and drifted to 2025 despite November 2026 source
dates. Web replies therefore use bounded verbatim source excerpts with links,
not unchecked generated summaries. Release listings render extracted titles and
dates/genres as a clean list, excluding site navigation, ratings and buttons. Direct searches open the first Google result;
other requests can select tools, but return quoted evidence once a page is read.
Failure to read sources cannot produce an answer from model memory. This scopes
web answer capability to source excerpts; arbitrary synthesis is not certified.
The current UTC date lives in every chat's runtime system prompt and refreshes
on a new day, including saved-chat replay, without editing visible messages.

Validation: release workspace build, 567 library tests passed (32 ignored), and
`lisa-bench smoke` passed after detecting 128 GiB physical RAM and a 120 GiB
maximum host budget. The ignored model-dependent cases remain unvalidated.
Desktop/mobile/offscreen WebKit extraction and Swift Markdown checks passed.
Debug Xcode builds passed for macOS, Simulator and the signed iPhone. The real
phone's original movie query passed Google extraction, source-page reading,
completed source excerpt, hidden-browser and no-memory-warning checks. A live
consent challenge was not encountered; Simulator inference retains the bf16
compiler blocker described above.

### Frontend cleanup — 2026-11-19

The retired AppKit frontend was removed and the shared Rust library moved from
`lisa-ui` to `lisa-app`. Its C ABI and persisted chat schema remain unchanged.
The SwiftUI release bundle is 14,493,120 bytes; its signature verifies, its dSYM
UUID matches the executable, and its kernel package matches the Rust build.
The former 12 MB packaging gate was removed with user approval.

The release workspace build, all 571 library tests (32 ignored), GPU smoke,
Markdown rendering checks and WebKit extraction checks passed. Debug builds
passed for macOS and the signed physical iPhone. LFM3 1.7B's real Mac app check
passed download/discovery, loading, committed replies, multi-turn context,
cancellation, saved-history replay and persisted model identity.
The physical iPhone passed those checks plus foreground reload/history replay
and memory-warning release without automatic reload. These remain functional
checks, not new numerical parity or performance measurements.

### Bonsai image generation — 2026-11-20

The signed physical iPhone 15 Pro Max (A17 Pro, iOS 27.2.1) downloaded and loaded
Bonsai Binary 4B, generated a 512×512 four-step observatory image, and saved a
440,169-byte PNG under Application Support/Lisa/generated-images. The file was
copied back and visually inspected. Generation took 199.55 seconds in one
foreground run with optimized Release Rust and the Swift DEBUG integration
hook enabled. No memory warning was reported. This is functional evidence,
not a throughput benchmark or full PrismML numerical-parity certification.

The live admission budget was 2,970,761,376 bytes after the 512 MiB reserve;
the model requires 2.6 GiB. Download size is independent of that working budget.
An earlier run completed compute but failed at the old HOME/Pictures output
path; the explicit application-storage path fixes that failure. Another run
was cancelled when the app became inactive, as expected for the then-current
foreground-only lifecycle. The successful run remained active throughout.

macOS and signed iOS builds passed. The release workspace build, 574 library
tests (32 ignored), and GPU smoke suite passed. Kernel, tiling and retained
FLUX image checks are detailed in [media support](media.md#bonsai-image-binary-4b).

### Continued-processing integration — 2026-11-20

macOS and the personal-team signed iPhone Release builds passed with the shared
background-generation coordinator. The physical iPhone check passed local replies,
multi-turn context, cancellation, persisted history, transient inactivity without
unloading, foreground reload, and memory-warning release. Lifecycle transitions
in this check were invoked directly; this does not validate OS background GPU use.
The runtime reported no available background GPU resource for this build/device.

Xcode explicitly rejected the GPU entitlement for the personal development team.
The user chose to retain that team, so the installed app omits the entitlement
and remains foreground-only. An eligible signed build still needs the real
background image check, system cancellation, and lock-screen completion checks.
The release workspace build, 574 library tests (32 ignored), and smoke suite passed.

### Generation activity

Text generation logs the user-observed output rate for ordinary chat,
image-assisted chat and each browser-agent pass, across model families. The
rate uses tokens after the first divided by time from the first to the last
emitted token. Time to first token is reported separately; it includes prefill
and first-token sampling. Counts include reasoning and control tokens but
exclude EOS. Fewer than two tokens have no meaningful decode rate. These
interactive measurements include application overhead and are not isolated
engine benchmarks.
