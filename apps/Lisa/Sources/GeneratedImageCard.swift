import SwiftUI
import QuickLook
import UniformTypeIdentifiers

struct GeneratedImageCard: View {
    let path: String
    private let image: Image?
    private let dimensions: String
    @State private var previewURL: URL?
    @State private var export: GeneratedImageFile?
    @State private var exporting = false
    @State private var error: String?

    init(path: String) {
        self.path = path
        #if os(macOS)
        let source = NSImage(contentsOfFile: path)
        image = source.map { Image(nsImage: $0) }
        let pixels = source?.representations.first { $0.pixelsWide > 0 && $0.pixelsHigh > 0 }
        dimensions = pixels.map { "\($0.pixelsWide) × \($0.pixelsHigh)" } ?? "Saved locally"
        #else
        let source = UIImage(contentsOfFile: path)
        image = source.map { Image(uiImage: $0) }
        dimensions = source.map { "\(Int($0.size.width * $0.scale)) × \(Int($0.size.height * $0.scale))" } ?? "Saved locally"
        #endif
    }

    private var url: URL { URL(fileURLWithPath: path) }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let image {
                Button(action: openImage) {
                    image.resizable().scaledToFit()
                        .frame(maxWidth: .infinity)
                        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                        .contentShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Open generated image")
                .help("Open full-size image")
            } else {
                Label("Image unavailable", systemImage: "photo")
                    .foregroundStyle(.secondary).padding(24)
            }
            HStack(spacing: 6) {
                Text(dimensions).font(.caption).foregroundStyle(.secondary)
                Spacer(minLength: 8)
                Button(action: openImage) {
                    Image(systemName: "arrow.up.left.and.arrow.down.right").frame(width: actionSize, height: actionSize)
                }.help("Open full-size image").accessibilityLabel("Open full-size image")
                Button(action: saveCopy) {
                    Image(systemName: "arrow.down.to.line").frame(width: actionSize, height: actionSize)
                }.help("Save image…").accessibilityLabel("Save image")
                #if os(macOS)
                Button { NSWorkspace.shared.activateFileViewerSelecting([url]) } label: {
                    Image(systemName: "folder").frame(width: actionSize, height: actionSize)
                }.help("Show in Finder").accessibilityLabel("Show in Finder")
                #endif
                ShareLink(item: url) {
                    Image(systemName: "square.and.arrow.up").frame(width: actionSize, height: actionSize)
                }.help("Share image").accessibilityLabel("Share image")
            }
            .buttonStyle(.borderless).tint(.primary)
            .padding(.horizontal, 8).padding(.top, 6)
        }
        .padding(8)
        .frame(maxWidth: 480, alignment: .leading)
        .background(cardBackground, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
        .quickLookPreview($previewURL)
        .fileExporter(isPresented: $exporting, document: export, contentType: .png,
                      defaultFilename: url.deletingPathExtension().lastPathComponent) { result in
            if case .failure(let failure) = result { error = failure.localizedDescription }
        }
        .alert("Image action failed", isPresented: Binding(get: { error != nil }, set: { if !$0 { error = nil } })) {
            Button("OK") { error = nil }
        } message: { Text(error ?? "") }
    }

    private var cardBackground: AnyShapeStyle {
        #if os(macOS)
        AnyShapeStyle(.quaternary.opacity(0.4))
        #else
        AnyShapeStyle(Color(uiColor: .secondarySystemBackground))
        #endif
    }

    private var actionSize: CGFloat {
        #if os(macOS)
        28
        #else
        44
        #endif
    }

    private func openImage() {
        #if os(macOS)
        if !NSWorkspace.shared.open(url) { error = "The image could not be opened. Use Show in Finder to locate the saved file." }
        #else
        previewURL = url
        #endif
    }

    private func saveCopy() {
        do {
            export = GeneratedImageFile(data: try Data(contentsOf: url))
            exporting = true
        } catch { self.error = error.localizedDescription }
    }
}

private struct GeneratedImageFile: FileDocument {
    static var readableContentTypes: [UTType] { [.png] }
    let data: Data
    init(data: Data) { self.data = data }
    init(configuration: ReadConfiguration) throws {
        guard let contents = configuration.file.regularFileContents else { throw CocoaError(.fileReadCorruptFile) }
        data = contents
    }
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}
