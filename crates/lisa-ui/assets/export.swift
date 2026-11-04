// Regenerate Lisa.png and Lisa.icns from the editable SVG using macOS only.
// Run: swift crates/lisa-ui/assets/export.swift
import AppKit

let assets = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let files = FileManager.default
let temporary = files.temporaryDirectory.appendingPathComponent(UUID().uuidString)
let iconset = temporary.appendingPathComponent("Lisa.iconset")
try files.createDirectory(at: iconset, withIntermediateDirectories: true)
defer { try? files.removeItem(at: temporary) }

guard let image = NSImage(contentsOf: assets.appendingPathComponent("Lisa.svg")) else {
    fatalError("Cannot load Lisa.svg")
}

func render(_ size: Int) throws -> Data {
    guard let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
    ), let context = NSGraphicsContext(bitmapImageRep: bitmap) else {
        fatalError("Cannot create icon bitmap")
    }
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = context
    context.imageInterpolation = .high
    image.draw(
        in: NSRect(x: 0, y: 0, width: size, height: size),
        from: .zero, operation: .copy, fraction: 1
    )
    context.flushGraphics()
    NSGraphicsContext.restoreGraphicsState()
    guard let png = bitmap.representation(using: .png, properties: [:]) else {
        fatalError("Cannot encode icon PNG")
    }
    return png
}

for size in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let suffix = scale == 2 ? "@2x" : ""
        let png = try render(size * scale)
        try png.write(to: iconset.appendingPathComponent("icon_\(size)x\(size)\(suffix).png"))
        if size * scale == 1024 {
            try png.write(to: assets.appendingPathComponent("Lisa.png"), options: .atomic)
        }
    }
}
let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
process.arguments = ["-c", "icns", iconset.path, "-o", assets.appendingPathComponent("Lisa.icns").path]
try process.run()
process.waitUntilExit()
guard process.terminationStatus == 0 else { fatalError("iconutil failed") }
let optimize = Process()
optimize.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
optimize.arguments = [assets.appendingPathComponent("optimize.py").path]
try optimize.run()
optimize.waitUntilExit()
guard optimize.terminationStatus == 0 else { fatalError("Icon compression failed") }
print("Exported Lisa.png and Lisa.icns")
