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
    var title: String {
        let value = turns.first?.user ?? draft
        return value.isEmpty ? "New chat" : String(value.prefix(60))
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
    static let defaultSystem = "You are Lisa, a helpful and knowledgeable AI assistant. Answer clearly and concisely."
    let bridge = NativeBridge()
    let browser = BrowserController()
    let speech = SpeechController()
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

    var current: Conversation? { chats.first { $0.id == activeID } }
    var canSend: Bool { ready && !busy && !switching && loadedModel != nil && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    func start() async {
        guard !started else { return }; started = true
        do {
            let support = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true).appendingPathComponent("Lisa")
            try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
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
                    } catch { self.error = error.localizedDescription; self.ready = false; self.busy = false; self.loadingModel = nil; self.loadedModel = nil; self.status = "Inference worker stopped"; return }
                    try? await Task.sleep(for: .milliseconds(60))
                }
            }
            #if DEBUG && targetEnvironment(simulator)
            await runSimulatorCheckIfRequested()
            #endif
        } catch { self.error = error.localizedDescription; status = "Lisa could not start" }
    }

    func refreshModels() async throws {
        let result = try await bridge.call(["op":"models"]) as? [String:Any] ?? [:]
        models = try bridge.decoded([ModelChoice].self, result["models"] ?? [])
        models.sort { if $0.local != $1.local { return $0.local }; return false }
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
        await save(); attachments = []; speech.stop()
        do {
            _ = try await bridge.call(["op":"clear"])
            let value = try await bridge.call(["op":"new_chat", "model":loadedModel as Any? ?? NSNull(), "system":Self.defaultSystem])
            let chat = try bridge.decoded(Conversation.self, value)
            chats.insert(chat, at: 0); activeID = chat.id; draft = ""; system = chat.system
            await save()
        } catch { self.error = error.localizedDescription }
    }
    func selectChat(_ id: String) async {
        guard !busy && !switching, id != activeID, let chat = chats.first(where: { $0.id == id }) else { return }
        switching = true; defer { switching = false }
        await save(); speech.stop(); attachments = []; activeID = id; draft = chat.draft; system = chat.system
        do {
            _ = try await bridge.call(["op":"restore", "turns":try bridge.object(chat.turns), "system":system])
            if let id = chat.model, id != loadedModel { await loadModel(id) }
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
        guard ready && !busy else { return }
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
        downloading = id; downloadProgress = 0
        do { _ = try await bridge.call(["op":"download", "id":id]) }
        catch { self.error = error.localizedDescription; downloading = nil }
    }
    func send() async {
        guard canSend else { return }
        speech.stop()
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        busy = true
        do {
            guard activeID != nil else { throw CocoaError(.fileNoSuchFile) }
            let kept = attachments
            updateCurrent { $0.turns.append(Turn(user:text, images:kept)) }
            do {
                _ = try await bridge.call(["op":"chat", "text":text, "system":system, "images":kept, "browse":webEnabled && !imageMode])
            } catch {
                updateCurrent { $0.turns.removeLast() }
                throw error
            }
            draft = ""; attachments = []; status = "Generating…"; await save()
        } catch { self.error = error.localizedDescription; busy = false }
    }
    func cancel() async {
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
        #if os(iOS)
        let leaving = phase != .active
        #else
        let leaving = phase == .background
        #endif
        if leaving && !backgrounded {
            backgrounded = true; restoreOnForeground = loadedModel != nil || loadingModel != nil
            speech.stop(); browser.cancel()
            #if os(iOS)
            suspending = true; busy = true
            do { _ = try await bridge.call(["op":"unload"]) }
            catch { self.error = error.localizedDescription; suspending = false; busy = false }
            #endif
            await save()
        } else if phase == .active, backgrounded {
            backgrounded = false
            #if os(iOS)
            if restoreOnForeground && !suspending && !busy, let id = current?.model { restoreOnForeground = false; await loadModel(id) }
            #endif
        }
    }
    private func receive(_ event: [String:Any]) async {
        let kind = event["kind"] as? String ?? ""
        let value = event["value"] ?? NSNull()
        switch kind {
        case "log":
            status = value as? String ?? status; logs.append(status); if logs.count > 500 { logs.removeFirst(logs.count - 500) }
        case "replace", "delta", "committed":
            guard let text = value as? String else { break }
            updateCurrent { chat in
                guard !chat.turns.isEmpty else { return }
                let i = chat.turns.count - 1
                if kind == "delta" { chat.turns[i].assistant += text } else { chat.turns[i].assistant = text }
                if kind == "committed" { chat.turns[i].complete = true }
            }
        case "image":
            updateCurrent { chat in if !chat.turns.isEmpty { chat.turns[chat.turns.count - 1].generated_image = value as? String } }
        case "done":
            busy = suspending; status = suspending ? "Releasing model…" : "Ready"; await save()
            if speakReplies, let turn = current?.turns.last, turn.complete { speech.speak(turn.assistant) }
        case "loaded":
            busy = false
            if value as? Bool == true { loadedModel = loadingModel; status = "Ready" }
            else { loadedModel = nil }
            loadingModel = nil
        case "unloaded":
            suspending = false; busy = false; loadedModel = nil; loadingModel = nil
            status = "Model released"
            #if os(iOS)
            if !backgrounded && restoreOnForeground, let id = current?.model {
                restoreOnForeground = false; await loadModel(id)
            }
            #endif
        case "supports_images": supportsImages = value as? Bool ?? false
        case "image_mode": imageMode = !(value is NSNull)
        case "serving": serverURL = value as? String
        case "download":
            if let progress = value as? [String:Any], let total = progress["total"] as? Double, total > 0 {
                downloadProgress = (progress["completed"] as? Double ?? 0) / total
            }
        case "models_changed":
            downloading = nil
            do { try await refreshModels() } catch { self.error = error.localizedDescription }
        case "browse":
            guard let request = value as? [String:Any] else { break }
            showBrowser = true
            browser.navigate(request) { [weak self] response in
                guard let self else { return }
                do { _ = try await self.bridge.call(response) } catch { self.error = error.localizedDescription }
            }
        default: break
        }
    }
}
