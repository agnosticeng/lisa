#!/bin/sh
# Build lisa-ui and package it as target/release/Lisa.app so the macOS menu bar
# shows "lisa" (a bare binary shows its filename instead).
set -e
cd "$(dirname "$0")/../.."
cargo build --release -p lisa-ui

APP=target/release/Lisa.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp target/release/lisa-ui "$APP/Contents/MacOS/lisa-ui"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>lisa</string>
  <key>CFBundleDisplayName</key><string>lisa</string>
  <key>CFBundleExecutable</key><string>lisa-ui</string>
  <key>CFBundleIdentifier</key><string>com.lisa.ui</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

echo "built $APP  (run: open $APP)"
