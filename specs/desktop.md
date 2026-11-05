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
