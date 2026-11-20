// Regenerate the full-bleed, opaque iOS icon from the shared editable artwork.
// Run from the repository root: swift apps/Lisa/generate-icon.swift
import AppKit

let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let source = directory.appendingPathComponent("assets/Lisa.svg")
let destination = directory.appendingPathComponent("Assets.xcassets/AppIcon.appiconset/AppIcon.png")
var svg = try String(contentsOf: source, encoding: .utf8)
// The Mac artwork includes its own inset tile and rim. iOS masks the entire
// image itself: use the tile's coordinate space and remove its baked-in corners.
let replacements = [
    ("viewBox=\"0 0 1024 1024\"", "viewBox=\"100 100 824 824\""),
    ("width=\"824\" height=\"824\" rx=\"186\"", "width=\"824\" height=\"824\""),
    ("  <rect x=\"102\" y=\"102\" width=\"820\" height=\"820\" rx=\"184\" fill=\"none\" stroke=\"url(#rim)\" stroke-width=\"3\"/>", "")
]
for (before, after) in replacements {
    precondition(svg.contains(before), "Shared icon geometry changed; update the iOS export")
    svg = svg.replacingOccurrences(of: before, with: after)
}
let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).svg")
try svg.write(to: temporary, atomically: true, encoding: .utf8)
defer { try? FileManager.default.removeItem(at: temporary) }
guard let image = NSImage(contentsOf: temporary),
      let bitmap = CGContext(data: nil, width: 1024, height: 1024, bitsPerComponent: 8,
        bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
    fatalError("Cannot render iOS icon")
}
let context = NSGraphicsContext(cgContext: bitmap, flipped: false)
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = context
context.imageInterpolation = .high
image.draw(in: NSRect(x: 0, y: 0, width: 1024, height: 1024), from: .zero,
    operation: .copy, fraction: 1)
context.flushGraphics()
NSGraphicsContext.restoreGraphicsState()
guard let rendered = bitmap.makeImage(),
      let png = NSBitmapImageRep(cgImage: rendered).representation(using: .png, properties: [:]) else {
    fatalError("Cannot encode iOS icon")
}
try png.write(to: destination, options: .atomic)
print("Exported full-bleed iOS AppIcon.png")
