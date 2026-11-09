# Desktop app

The native AppKit app lives in [lisa-ui](../crates/lisa-ui/src/). Launch
`lisa-ui` directly or build the signed local bundle with
`sh crates/lisa-ui/bundle.sh`. The editable icon and generated assets live in
[assets](../crates/lisa-ui/assets/).

Metal inference requires macOS 15 or later, including on M1–M4 Macs. The
compiler's separate math controls and MSL 3.2 are unavailable on macOS 14.
The bundle declares this minimum, and standalone launches show an upgrade
message before starting the engine. Shared model loading and runtime creation
also reject unsupported macOS versions before allocating GPU resources. This
keeps the existing arithmetic contracts; an older-OS compiler fallback would
need separate numerical validation.

The bundle script produces `target/dist/Lisa.app` and enforces a 10,000,000-byte
limit for the complete bundle, including the engine, UI, kernel package, icon
and notices. The `dist` profile uses full LTO; engine/runtime optimization stays
at level 3 while UI and template rendering favor size. Only the bundled copy
is stripped. Keep the matching `target/dist/lisa-ui.dSYM` and unstripped binary
for symbolication and source-level debugging. Model weights are separate.

## Navigation and selection

The sidebar contains **New chat**, **Chats** and **Models**. Chats and Models
collapse independently, with the preference retained locally. Disclosure
controls appear on hover. Downloaded models sort before catalog-only entries.

There is one active chat ID. Selection is derived from it when rows are rebuilt;
hover uses each row's actual pointer bounds. New chat is an action and only has
a hover background. Model selection follows the active model identity after
list sorting. Chat/model icons and padded row backgrounds should remain visually
consistent.

Models use a vertical stack of native clickable rows in a flipped scroll
document. Bottom clearance belongs to that document, so the last row and its
subtitle can be brought fully into view. Selection, loading and download
updates preserve the scroll position. Resizing preserves the bottom position
when the list is scrolled to the end, and otherwise retains the current offset.

Settings and Activity buttons have been removed from the sidebar. **Settings**
in the chat header and **View > Model Settings** open the inspector;
**View > Toggle Logs Panel** (Control-Command-L) opens logs. Window/sidebar
resizing preserves the user's inspector visibility choice. Logs use the shared
workspace background.

These are the intended interaction invariants as well as the current code
structure. Future sidebar edits need focused checks for stale hover after
scrolling/rebuilding, exclusive selection, collapse layout and resize behavior.

## Saved chats

[Chat storage](../crates/lisa-ui/src/chats.rs) persists messages, drafts, model
identity and copied input attachments under
`~/Library/Application Support/Lisa/chats`. Reopening a chat restores its
content and selects/loads its model as needed. Text history can be continued
after an app restart; this does not imply that GPU cache state survives restart.

Chat deletion is available from the row action and menu, with confirmation.
The app removes the chat record and owned attachment copies; it reports partial
cleanup failures in the log. The active chat is replaced safely when deleted.
**New chat** (Command-N) starts a separate conversation. Return sends and
Shift-Return inserts a line break.

Every language-model chat turn gives the model access to the `web_search` and
`web_open` tools. It can search for sources, read page text and follow links
before answering. The model chooses when to browse from ordinary messages;
there is no browser slash command or keyword-triggered navigation.
Image-generation turns do not use the language-model agent.
Lisa loads normal web pages in a visible WebKit browser panel and extracts
readable text and links from the rendered page; it does not send whole HTML documents to the model or use a search API key.
If Google shows consent or verification, Lisa asks in the chat for the user to
review and resolve the prompt in the visible browser. It does not accept
cookies or solve verification automatically. The pending navigation waits up
to five minutes, then reports a timeout; it resumes after the prompt is gone.

Each turn has a budget of six browser actions, executed sequentially in the
same WebKit view. Multiple tool calls each receive a result; invalid arguments
and navigation failures are returned to the model so it can correct or report
them. Navigation callbacks prevent a failed load from returning stale page
content. A page read returns up to 12,000 characters and 24 links, preferring
main/article content. Search returns up to eight result cards. Before each
model pass, long excerpts are shortened to fit its context limit while keeping
page headers and trailing links; older complete chat pairs are dropped only
when needed. The current question and tool-call associations remain intact. Sources added by
the UI come only from opened pages, and Markdown source links are clickable.
Answer text streams during generation. Partial control markers, reasoning
and tool arguments are withheld; final tool parsing and source formatting
settle the completed reply.

Extraction depends on the page DOM and can fail if a site blocks access or
changes its page structure. Google verification challenges require manual
input. Closing the browser panel hides it; the next tool navigation shows it
again. Search queries and requested URLs reach Google and the visited sites
through normal browser requests; model inference remains local.

November 9 integration validation: release workspace build; 356 library tests
passed (26 existing tests ignored); eight UI tests passed; kernel smoke passed.
The extraction scripts passed JavaScript syntax and Google prompt detection
checks; a fresh consent prompt was not exercised live. The signed dist bundle
is 9,895,697 bytes, with matching symbols outside the app. The UI crate uses
size optimization `z`; inference crates retain their existing settings.

Live UI checks confirmed the slash menu has no `/web` command, WebKit searches
execute, and `web_open` reads example.com and produces a clickable source.
The final 0.8B model run reported a 262,144-token metadata context in the UI
log. Short CLI runs passed with the Qwen 27B GGUF metadata default and an
explicit 512-token override. These checks do not certify long-context accuracy
or reliable research: the 27B search run exhausted its action budget without
opening a source, and the 0.8B model failed to browse for a Rust follow-up.
The streaming regression check observed the 0.8B reply grow while Lisa was
responding, then return to Ready. The response repeated itself until its token
limit; this verifies delivery behavior, not answer quality. Boundary tests
cover split tool markers, hidden reasoning and Unicode text.

## Models and downloads

The allowlist is [catalog.rs](../crates/lisa-ui/src/catalog.rs), with seven
language checkpoints and three image generators. Hub search does not add new
catalog entries. Existing compatible local checkpoints remain discoverable.
Type icons distinguish image generators; download actions use an icon.

[Discovery](../crates/lisa-ui/src/models.rs) shares checkpoint identities and
snapshot ordering with the engine. It validates available package/shard files
before listing a local model and excludes decision-only Laya checkpoints from
chat. Local size accounting follows Hub links, counts each inode once and
avoids directory cycles. The displayed RAM budget uses the engine's shared
budget calculation.

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

November 9 downloader validation: release workspace build, 358 library tests
(26 existing tests ignored), eight UI tests and kernel smoke passed. Local Hub
fixtures checked partial-transfer resumption, interrupted package discovery,
cache reuse, truncated-file repair, nested files and stale-token fallback.
A fresh 12,512,611-byte public repository download passed with expired ambient
credentials. The lighter client's signed workspace bundle is 11,296,700 bytes;
it fits the 12 MB packaging limit adopted for the synchronous Hub client.

The list separates download size from resident memory estimates. DeepSeek's
Engram table is disk-only. GGUF packages select the required files and skip
legacy MLX preparation. Original model licenses/notices remain with downloads.
The catalog's memory check is an estimate; runtime admission still accounts for
caches and scratch. Only one model is loaded at a time.

Text chat and the app server use automatic MTP where the checkpoint supports
it. This enables GLM's embedded head; it does not create an absent DeepSeek
head or prove a speedup. Refer to [model status](README.md#model-status).

## Images

Attach is enabled for supported vision/text models. Message thumbnails retain
their own images. Selecting an image generator enables Generate and disables
Attach; default steps come from the loaded pipeline.

Generated PNGs are saved under `~/Pictures/Lisa` and displayed as image cards,
with Open and Show in Finder actions. A raw saved-file path is not the main
assistant response. The [README preview](../docs/images/screenshot.png) shows
an actual local FLUX.2 Klein 9B generation in the current app, framed at 16:10.
It is a product preview, not a performance measurement.

Detailed inference limits and integration evidence are in [media](media.md).
