import Foundation
import SwiftUI

struct Turn: Codable {
    var user: String
    var assistant = ""
    var images: [String] = []
    var generated_image: String?
    var complete = false
}
struct Conversation: Codable, Identifiable {
    var id: String
    var model: String?
    var system: String
    var updated: UInt64
    var turns: [Turn]
    var draft: String
    var hasContent: Bool { !turns.isEmpty || !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    var title: String {
        let value = (turns.first?.user ?? draft).trimmingCharacters(in: .whitespacesAndNewlines)
        let firstLine = value.split(whereSeparator: \.isNewline).first.map(String.init) ?? ""
        return firstLine.isEmpty ? "New chat" : String(firstLine.prefix(60))
    }
}
struct ModelChoice: Decodable, Identifiable {
    let id: String
    let name: String
    let detail: String
    let size: UInt64
    let local: Bool
    let path: String?
    let fits: Bool
    var sizeLabel: String { ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file) }
}

/// The C boundary receives owned bytes. Filesystem work never blocks the UI thread.
@MainActor final class NativeBridge {
    private let queue = DispatchQueue(label: "lisa.app.bridge", qos: .userInitiated)
    func call(_ request: [String: Any]) async throws -> Any {
        let input = try JSONSerialization.data(withJSONObject: request)
        let response: Data = try await withCheckedThrowingContinuation { continuation in
            queue.async {
                let string = String(decoding: input, as: UTF8.self)
                let pointer = string.withCString { lisa_app_request($0) }
                guard let pointer else {
                    continuation.resume(throwing: NSError(domain: "Lisa", code: 1, userInfo: [NSLocalizedDescriptionKey: "The application bridge returned no response"]))
                    return
                }
                let bytes = Data(bytes: pointer, count: strlen(pointer))
                lisa_app_free(pointer)
                continuation.resume(returning: bytes)
            }
        }
        let envelope = try JSONSerialization.jsonObject(with: response) as? [String: Any] ?? [:]
        guard envelope["ok"] as? Bool == true else {
            throw NSError(domain: "Lisa", code: 1, userInfo: [NSLocalizedDescriptionKey: envelope["error"] as? String ?? "Application request failed"])
        }
        return envelope["value"] ?? NSNull()
    }
    func decoded<T: Decodable>(_ type: T.Type, _ value: Any) throws -> T {
        try JSONDecoder().decode(type, from: JSONSerialization.data(withJSONObject: value, options: .fragmentsAllowed))
    }
    func object<T: Encodable>(_ value: T) throws -> Any {
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
    }
}

@MainActor final class AppStore: ObservableObject {
    static let defaultModel = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
    static let defaultSystem = "You are Lisa, a helpful and knowledgeable AI assistant. Answer clearly and concisely."
    let bridge = NativeBridge()
    let browser = BrowserController()
    let speech = SpeechController()
    let backgroundGeneration = BackgroundGeneration()
    private var generationSucceeded = false
    private var generationCancelled = false
    @Published var chats: [Conversation] = []
    @Published var activeID: String?
    @Published var models: [ModelChoice] = []
    @Published var draft = ""
    @Published var system = defaultSystem
    @Published var attachments: [String] = []
    @Published var status = "Starting Lisa…"
    @Published var error: String?
    @Published var busy = false
    @Published var ready = false
    @Published var loadedModel: String?
    @Published var loadingModel: String?
    @Published var downloading: String?
    @Published var downloadProgress: Double = 0
    @Published var supportsImages = false
    @Published var imageMode = false
    @Published var serverURL: String?
    @Published var logs: [String] = []
    @Published var showBrowser = false
    @Published var switching = false
    @AppStorage("lisa.webEnabled") var webEnabled = true
    @AppStorage("lisa.speakReplies") var speakReplies = false
    private var polling: Task<Void, Never>?
    private var draftSave: Task<Void, Never>?
    private var started = false
    private var backgrounded = false
    private var suspending = false
    private var restoreOnForeground = false
    private var memoryPressure = false
    private var loadAfterDownload: String?

    var visibleChats: [Conversation] { chats.filter(\.hasContent) }
    var current: Conversation? { chats.first { $0.id == activeID } }
    var canSend: Bool { ready && !busy && !switching && !backgrounded && !suspending && loadedModel != nil && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    func start() async {
        guard !started else { return }; started = true
        backgroundGeneration.onLog = { [weak self] in self?.logs.append($0) }
        backgroundGeneration.onExpiration = { [weak self] in
            Task { @MainActor in
                guard let self else { return }
                self.error = "iOS stopped the background task. You can retry the request."
                await self.releaseForBackground()
            }
        }
        do {
            let support = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true).appendingPathComponent("Lisa")
            try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
            #if os(iOS)
            // Keep downloaded weights in this sandbox across launches, but do
            // not upload reproducible model files into the user's iCloud backup.
            var modelCache = support.appendingPathComponent("models", isDirectory: true)
            try FileManager.default.createDirectory(at: modelCache, withIntermediateDirectories: true)
            var cacheValues = URLResourceValues()
            cacheValues.isExcludedFromBackup = true
            try modelCache.setResourceValues(cacheValues)
            #endif
            _ = try await bridge.call(["op":"initialize", "storage":support.path])
            let saved = try await bridge.call(["op":"chats"]) as? [String:Any] ?? [:]
            chats = try bridge.decoded([Conversation].self, saved["chats"] ?? [])
            if let warnings = saved["errors"] as? [String], !warnings.isEmpty { error = warnings.joined(separator: "\n") }
            ready = true
            status = "Choose a model to get started"
            try await refreshModels()
            if let chat = chats.first { activeID = chat.id; draft = chat.draft; system = chat.system }
            else { await newChat() }
            polling = Task { [weak self] in
                while !Task.isCancelled {
                    guard let self else { return }
                    do {
                        let events = try await self.bridge.call(["op":"poll"]) as? [[String:Any]] ?? []
                        for event in events { await self.receive(event) }
                    } catch { self.error = error.localizedDescription; self.ready = false; self.busy = false; self.loadingModel = nil; self.loadedModel = nil; self.status = "Inference worker stopped"; self.backgroundGeneration.finish(success: false); return }
                    try? await Task.sleep(for: .milliseconds(60))
                }
            }
            #if DEBUG
            if await runIntegrationCheckIfRequested() { return }
            #endif
            await loadPreferredModel()
        } catch { self.error = error.localizedDescription; status = "Lisa could not start" }
    }

    func refreshModels() async throws {
        let result = try await bridge.call(["op":"models"]) as? [String:Any] ?? [:]
        let choices = try bridge.decoded([ModelChoice].self, result["models"] ?? [])
        models = choices.enumerated().sorted {
            if $0.element.fits != $1.element.fits { return $0.element.fits }
            if $0.element.local != $1.element.local { return $0.element.local }
            return $0.offset < $1.offset
        }.map(\.element)
    }

    private func loadPreferredModel() async {
        let id = current?.model ?? Self.defaultModel
        if models.contains(where: { $0.id == id && $0.local }) {
            await loadModel(id)
        } else if id == Self.defaultModel, models.contains(where: { $0.id == id && $0.fits }) {
            loadAfterDownload = id
            await download(id)
        } else {
            error = "The conversation’s model is unavailable on this device. Open Models to choose one."
        }
    }

    private func loadDownloadedModelIfReady() async {
        guard !busy && !backgrounded && !memoryPressure,
              let id = loadAfterDownload, models.contains(where: { $0.id == id && $0.local }) else { return }
        loadAfterDownload = nil
        await loadModel(id)
    }

    private func updateCurrent(_ edit: (inout Conversation) -> Void) {
        guard let i = chats.firstIndex(where: { $0.id == activeID }) else { return }
        edit(&chats[i]); chats[i].updated = UInt64(Date().timeIntervalSince1970 * 1_000_000_000)
    }
    func save() async {
        guard ready else { return }
        updateCurrent { $0.draft = draft; $0.system = system }
        guard let chat = current else { return }
        do { _ = try await bridge.call(["op":"save_chat", "chat":try bridge.object(chat)]) }
        catch { self.error = error.localizedDescription }
    }
    func scheduleSave() {
        draftSave?.cancel()
        draftSave = Task { [weak self] in
            do { try await Task.sleep(for: .milliseconds(350)) } catch { return }
            await self?.save()
        }
    }
    func newChat() async {
        guard !busy && !switching else { return }; switching = true; defer { switching = false }
        if current?.turns.isEmpty == true && draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            draft = ""; attachments = []; speech.stop()
            return
        }
        await save(); attachments = []; speech.stop()
        do {
            _ = try await bridge.call(["op":"clear"])
            let value = try await bridge.call(["op":"new_chat", "model":loadedModel ?? Self.defaultModel, "system":Self.defaultSystem])
            let chat = try bridge.decoded(Conversation.self, value)
            chats.insert(chat, at: 0); activeID = chat.id; draft = ""; system = chat.system
            await save()
        } catch { self.error = error.localizedDescription }
    }
    func selectChat(_ id: String) async {
        guard !busy && !switching, let chat = chats.first(where: { $0.id == id }) else { return }
        if id == activeID {
            if (chat.model ?? Self.defaultModel) != loadedModel { await loadPreferredModel() }
            return
        }
        switching = true; defer { switching = false }
        await save(); speech.stop(); attachments = []; activeID = id; draft = chat.draft; system = chat.system
        do {
            _ = try await bridge.call(["op":"restore", "turns":try bridge.object(chat.turns), "system":system])
            if (chat.model ?? Self.defaultModel) != loadedModel { await loadPreferredModel() }
        } catch { self.error = error.localizedDescription }
    }
    func deleteChat(_ id: String) async {
        guard !busy && !switching else { return }
        do {
            let warning = try await bridge.call(["op":"delete_chat", "id":id])
            if let warning = warning as? String { error = warning }
            chats.removeAll { $0.id == id }
            if activeID == id { activeID = nil; draft = ""; await newChat() }
        } catch { self.error = error.localizedDescription }
    }
    func loadModel(_ id: String) async {
        guard ready && !busy && !backgrounded && !suspending else { return }
        loadAfterDownload = nil
        memoryPressure = false
        busy = true; loadingModel = id; supportsImages = false; status = "Loading model…"
        do {
            // Restore persisted history before loading, including on the first launch.
            if let chat = current {
                _ = try await bridge.call(["op":"restore", "turns":try bridge.object(chat.turns), "system":system])
            }
            _ = try await bridge.call(["op":"load", "id":id])
            updateCurrent { $0.model = id }; await save()
        } catch { self.error = error.localizedDescription; busy = false; loadingModel = nil }
    }
    func download(_ id: String) async {
        guard downloading == nil else { return }
        error = nil
        downloading = id; downloadProgress = 0
        do { _ = try await bridge.call(["op":"download", "id":id]) }
        catch { self.error = error.localizedDescription; downloading = nil; loadAfterDownload = nil }
    }
    func send() async {
        guard canSend else { return }
        speech.stop()
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        busy = true
        generationSucceeded = false; generationCancelled = false
        await backgroundGeneration.begin(image: imageMode)
        if generationCancelled { busy = suspending; backgroundGeneration.finish(success: false); return }
        do {
            guard !suspending && (!backgrounded || backgroundGeneration.isActive) else {
                throw NSError(domain: "Lisa", code: 1, userInfo: [NSLocalizedDescriptionKey:"Keep Lisa open: background GPU execution is unavailable for this request."])
            }
            guard activeID != nil else { throw CocoaError(.fileNoSuchFile) }
            let kept = attachments
            let model = loadedModel
            updateCurrent { $0.model = model; $0.turns.append(Turn(user:text, images:kept)) }
            do {
                _ = try await bridge.call(["op":"chat", "text":text, "system":system, "images":kept, "browse":webEnabled && !imageMode])
            } catch {
                updateCurrent { $0.turns.removeLast() }
                throw error
            }
            draft = ""; attachments = []; status = "Generating…"; await save()
        } catch { self.error = error.localizedDescription; busy = suspending; backgroundGeneration.finish(success: false) }
    }
    func cancel() async {
        generationCancelled = true
        backgroundGeneration.cancelAdmission()
        browser.cancel(); speech.stop()
        do { _ = try await bridge.call(["op":"cancel"]) } catch { self.error = error.localizedDescription }
    }
    func setServer(_ enabled: Bool, address: String) async {
        do { _ = try await bridge.call(enabled ? ["op":"serve", "address":address] : ["op":"stop_server"]) }
        catch { self.error = error.localizedDescription }
    }
    func importImages(_ urls: [URL]) async {
        guard let id = activeID else { return }
        var accessible: [URL] = []
        defer { for url in accessible { url.stopAccessingSecurityScopedResource() } }
        for url in urls { if url.startAccessingSecurityScopedResource() { accessible.append(url) } }
        do {
            let kept = try await bridge.call(["op":"keep_images", "id":id, "paths":urls.map(\.path)]) as? [String] ?? []
            attachments.append(contentsOf: kept)
        } catch { self.error = error.localizedDescription }
    }
    func sceneChanged(_ phase: ScenePhase) async {
        guard ready else { return }
        logs.append("Scene changed: \(phase)")
        // Transient inactive states (system controls, interruptions) are not backgrounding.
        let leaving = phase == .background
        if leaving && !backgrounded {
            backgrounded = true
            restoreOnForeground = !memoryPressure && (loadedModel != nil || loadingModel != nil)
            speech.stop()
            #if os(iOS)
            if !backgroundGeneration.isActive { await releaseForBackground() }
            #else
            browser.cancel()
            #endif
            await save()
        } else if phase == .active, backgrounded {
            backgrounded = false
            #if os(iOS)
            if restoreOnForeground && !suspending && !busy, let id = current?.model { restoreOnForeground = false; await loadModel(id) }
            #endif
            await loadDownloadedModelIfReady()
        }
    }
    private func releaseForBackground() async {
        #if os(iOS)
        guard !suspending else { return }
        restoreOnForeground = !memoryPressure && (loadedModel != nil || loadingModel != nil)
        browser.cancel(); speech.stop()
        suspending = true; busy = true
        do { _ = try await bridge.call(["op":"unload"]) }
        catch {
            self.error = error.localizedDescription; suspending = false; busy = false
            backgroundGeneration.finish(success: false)
        }
        await save()
        #endif
    }
    #if os(iOS)
    /// Release through the serial worker so in-flight GPU users finish before
    /// their storage is freed. Memory pressure never triggers an automatic reload.
    func memoryWarning() async {
        guard ready else { return }
        logs.append("iOS memory warning: releasing the model")
        memoryPressure = true
        loadAfterDownload = nil
        restoreOnForeground = false
        speech.stop(); browser.cancel()
        guard !suspending else { return }
        suspending = true; busy = true; status = "Releasing model to free memory…"
        do { _ = try await bridge.call(["op":"unload"]) }
        catch { self.error = error.localizedDescription; suspending = false; busy = false }
        await save()
    }
    #endif
    private func receive(_ event: [String:Any]) async {
        let kind = event["kind"] as? String ?? ""
        let value = event["value"] ?? NSNull()
        switch kind {
        case "log":
            status = value as? String ?? status; logs.append(status); if logs.count > 500 { logs.removeFirst(logs.count - 500) }
        case "replace", "delta", "committed":
            guard let text = value as? String else { break }
            backgroundGeneration.update(text, delta: kind == "delta")
            updateCurrent { chat in
                guard !chat.turns.isEmpty else { return }
                let i = chat.turns.count - 1
                if kind == "delta" { chat.turns[i].assistant += text } else { chat.turns[i].assistant = text }
                if kind == "committed" { chat.turns[i].complete = true }
            }
        case "image":
            updateCurrent { chat in if !chat.turns.isEmpty { chat.turns[chat.turns.count - 1].generated_image = value as? String } }
        case "done":
            generationSucceeded = current?.turns.last?.complete == true
            busy = suspending; status = suspending ? "Releasing model…" : "Ready"; await save()
            if backgrounded && !suspending { await releaseForBackground() }
            if !suspending { backgroundGeneration.finish(success: generationSucceeded) }
            if !backgrounded && speakReplies, let turn = current?.turns.last, turn.complete { speech.speak(turn.assistant) }
        case "loaded":
            busy = false
            if value as? Bool == true { loadedModel = loadingModel; status = "Ready" }
            else { loadedModel = nil; error = logs.last ?? "The model could not be loaded. See Activity for details." }
            loadingModel = nil
        case "unloaded":
            backgroundGeneration.finish(success: generationSucceeded)
            suspending = false; busy = false; loadedModel = nil; loadingModel = nil
            supportsImages = false; imageMode = false
            status = "Model released"
            do { try await refreshModels() } catch { self.error = error.localizedDescription }
            #if os(iOS)
            if !backgrounded && restoreOnForeground, let id = current?.model {
                restoreOnForeground = false; await loadModel(id)
            }
            #endif
            await loadDownloadedModelIfReady()
        case "supports_images": supportsImages = value as? Bool ?? false
        case "image_mode": imageMode = !(value is NSNull)
        case "serving": serverURL = value as? String
        case "download":
            if let progress = value as? [String:Any], let total = progress["total"] as? Double, total > 0 {
                downloadProgress = (progress["completed"] as? Double ?? 0) / total
            }
        case "models_changed":
            do { try await refreshModels() } catch { self.error = error.localizedDescription }
            downloading = nil
            await loadDownloadedModelIfReady()
        case "download_failed":
            downloading = nil
            loadAfterDownload = nil
            error = value as? String ?? "Model download failed"
        case "browse":
            guard let request = value as? [String:Any] else { break }
            browser.navigate(request) { [weak self] response in
                guard let self else { return }
                if let message = response["error"] as? String {
                    self.showBrowser = false
                    self.logs.append("Browser failed: \(message)")
                } else if let message = response["needs_input"] as? String {
                    self.showBrowser = true
                    self.logs.append("Browser needs input: \(message)")
                } else {
                    self.showBrowser = false
                    let kind = response["search"] as? Bool == true ? "search" : "page"
                    self.logs.append("Browser \(kind) read: \(self.browser.address)")
                }
                do { _ = try await self.bridge.call(response) } catch { self.error = error.localizedDescription }
            }
        default: break
        }
    }
}
