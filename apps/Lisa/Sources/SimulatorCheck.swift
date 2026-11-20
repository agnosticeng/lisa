#if DEBUG
import Foundation
import Metal
#if os(iOS)
import UIKit
#endif

/// Opt-in integration check in the actual Apple app process, using its normal
/// AppStore, C bridge, engine thread, Metal runtime and persistent chat store.
/// No mock replies or remote inference are used. Results live in Documents.
@MainActor extension AppStore {
    private var checkPlatform: String {
        #if targetEnvironment(simulator)
        return "iOS Simulator"
        #elseif os(iOS)
        return "iOS device"
        #else
        return "macOS"
        #endif
    }
    func runIntegrationCheckIfRequested() async -> Bool {
        let arguments = ProcessInfo.processInfo.arguments
        if arguments.contains("--lisa-metal-probe") { await runMetalProbe(); return true }
        let legacy = arguments.contains("--lisa-simulator-check")
        let webCheck = arguments.contains("--lisa-web-check")
        guard legacy || webCheck || arguments.contains("--lisa-integration-check") else { return false }
        #if os(iOS)
        let previousIdleTimer = UIApplication.shared.isIdleTimerDisabled
        UIApplication.shared.isIdleTimerDisabled = true
        defer { UIApplication.shared.isIdleTimerDisabled = previousIdleTimer }
        #endif
        let previousWeb = webEnabled, previousSpeech = speakReplies
        defer { webEnabled = previousWeb; speakReplies = previousSpeech }
        let modelArgument = arguments.firstIndex(of: "--lisa-check-model").map { $0 + 1 }
        let model = modelArgument.flatMap { arguments.indices.contains($0) ? arguments[$0] : nil }
            ?? "mlx-community/Qwen3.5-0.8B-MLX-4bit"
        let resultURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(webCheck ? "web-check.json" : (legacy ? "simulator-check.json" : "integration-check.json"))
        var report: [String:Any] = ["model":model, "platform":checkPlatform, "os":ProcessInfo.processInfo.operatingSystemVersionString, "status":"running", "checks":[String](), "started":ISO8601DateFormatter().string(from: Date())]
        func record(_ stage: String) throws {
            report["stage"] = stage
            report["logs"] = Array(logs.suffix(30))
            report["chat_id"] = activeID
            report["assistant"] = current?.turns.last?.assistant ?? ""
            report["background_gpu_active"] = backgroundGeneration.isActive
            report["browser"] = ["address":browser.address, "status":browser.status, "shown":showBrowser,
                                 "loading":browser.webView.isLoading, "url":browser.webView.url?.absoluteString ?? ""]
            try FileManager.default.createDirectory(at: resultURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: resultURL, options: .atomic)
        }
        func check(_ condition: Bool, _ message: String) throws {
            if !condition { throw NSError(domain: "LisaIntegrationCheck", code: 1, userInfo: [NSLocalizedDescriptionKey:message]) }
        }
        func wait(_ timeout: TimeInterval, until condition: () -> Bool) async throws {
            let end = Date().addingTimeInterval(timeout)
            var nextReport = Date()
            while !condition() {
                if let error { throw NSError(domain: "LisaIntegrationCheck", code: 2, userInfo: [NSLocalizedDescriptionKey:error]) }
                try check(Date() < end, "Timed out: \(report["stage"] ?? "unknown") — \(status)")
                if Date() >= nextReport {
                    report["detail"] = status
                    report["download_progress"] = downloadProgress
                    try record(report["stage"] as? String ?? "unknown")
                    nextReport = Date().addingTimeInterval(5)
                }
                try await Task.sleep(for: .milliseconds(200))
            }
        }
        func passed(_ name: String) {
            var checks = report["checks"] as? [String] ?? []; checks.append(name); report["checks"] = checks
        }
        do {
            try record("download")
            if !models.contains(where: { $0.id == model && $0.local }) {
                await download(model)
                try await wait(900) { downloading == nil }
                try check(models.contains { $0.id == model && $0.local }, "Model download failed: \(status)")
            }
            passed("model downloaded")
            let replayIndex = arguments.firstIndex(of: "--lisa-web-replay").map { $0 + 1 }
            let replayID = replayIndex.flatMap { arguments.indices.contains($0) ? arguments[$0] : nil }
            let replayChat = replayID.flatMap { id in chats.first { $0.id == id } }
            await newChat(); webEnabled = false; speakReplies = false
            if webCheck, let replayChat, let index = chats.firstIndex(where: { $0.id == activeID }) {
                chats[index].turns = replayChat.turns.filter(\.complete)
                chats[index].system = replayChat.system
                system = replayChat.system
                _ = try await bridge.call(["op":"restore", "turns":try bridge.object(chats[index].turns), "system":system])
            }
            var memory = try await bridge.call(["op":"models"]) as? [String:Any] ?? [:]
            memory.removeValue(forKey: "models"); report["memory"] = memory
            try record("load")
            await loadModel(model)
            try await wait(300) { !busy }
            try check(loadedModel == model, "Model did not load: \(error ?? status)")
            passed("Metal model loaded")
            if webCheck {
                webEnabled = true
                try record("web search")
                let queryIndex = arguments.firstIndex(of: "--lisa-web-query").map { $0 + 1 }
                draft = queryIndex.flatMap { arguments.indices.contains($0) ? arguments[$0] : nil }
                    ?? "Search the internet for the latest movie releases and cite the source."
                await send()
                try await wait(360) { !busy }
                try check(current?.turns.last?.complete == true, "Web reply did not complete: \(current?.turns.last?.assistant ?? status)")
                try check(logs.contains { $0.contains("Searching the web") }, "The model answered without searching")
                try check(logs.contains { $0.hasPrefix("Browser search read:") }, "No readable search results were returned")
                try check(logs.contains { $0.hasPrefix("Browser page read:") }, "No Google result page was read")
                try check(!showBrowser, "Browser remained visible after reading")
                try check(!logs.contains { $0.contains("iOS memory warning") }, "Web search triggered a memory warning")
                passed("web search and reply completed")
                report["status"] = "passed"; report["chat_id"] = activeID
                try record("complete")
                return true
            }
            if arguments.contains("--lisa-image-check") {
                try record("image generation")
                draft = "A tiny warm-lit observatory on a mountain beneath violet stars, cinematic illustration, no text."
                let started = Date()
                await send()
                if arguments.contains("--lisa-background-check") {
                    try check(backgroundGeneration.isActive, "Background GPU authorization unavailable")
                    passed("background GPU authorized")
                }
                try await wait(900) { !busy }
                if arguments.contains("--lisa-background-check") {
                    try check(logs.contains("Scene changed: background"), "App never entered the background")
                    passed("generation survived real background transition")
                }
                try check(current?.turns.last?.complete == true, "Image generation did not complete")
                guard let path = current?.turns.last?.generated_image else {
                    throw NSError(domain: "LisaIntegrationCheck", code: 3, userInfo: [NSLocalizedDescriptionKey:"No generated image was saved"])
                }
                let data = try Data(contentsOf: URL(fileURLWithPath: path))
                try check(data.starts(with: [137,80,78,71,13,10,26,10]), "Generated file is not PNG")
                report["image_bytes"] = data.count; report["image_path"] = path
                report["generation_seconds"] = Date().timeIntervalSince(started)
                passed("local image generated and saved")
                try check(!logs.contains { $0.contains("iOS memory warning") }, "Image generation triggered a memory warning")
                passed("no memory warning")
                report["status"] = "passed"; try record("complete")
                return true
            }
            let savedID = activeID!
            try record("first reply")
            draft = "Remember that the code word for this conversation is violet. Reply with only READY."
            await send(); try await wait(180) { !busy }
            try check(current?.turns.last?.complete == true, "First generation failed: \(current?.turns.last?.assistant ?? status)")
            report["first_reply"] = current?.turns.last?.assistant ?? ""
            passed("first reply committed")
            try record("follow-up")
            draft = "What is the code word I asked you to remember? Reply with only the word."
            await send(); try await wait(180) { !busy }
            let answer = current?.turns.last?.assistant ?? ""
            report["follow_up"] = answer
            try check(current?.turns.last?.complete == true && answer.lowercased().contains("violet"), "Follow-up did not recall the code word: \(answer)")
            passed("multi-turn context")
            try record("cancellation")
            draft = "Write a very long numbered list of 500 different animals, one per line. Begin now and continue until number 500."
            await send()
            try await wait(180) { (current?.turns.last?.assistant.count ?? 0) > 30 || !busy }
            try check(busy, "Generation finished before cancellation could be tested")
            await cancel(); try await wait(60) { !busy }
            try check(current?.turns.last?.complete == false, "Cancelled output was committed to replay history")
            passed("cancellation leaves an incomplete turn")
            try record("saved-chat replay")
            await newChat(); await selectChat(savedID)
            draft = "What was our code word? Reply with only the word."
            await send(); try await wait(180) { !busy }
            let replay = current?.turns.last?.assistant ?? ""
            report["replayed_reply"] = replay
            try check(current?.turns.last?.complete == true && replay.lowercased().contains("violet"), "Saved-chat replay failed: \(replay)")
            passed("saved-chat replay excludes cancelled turn")
            await save()
            let saved = try await bridge.call(["op":"chats"]) as? [String:Any] ?? [:]
            let records = try bridge.decoded([Conversation].self, saved["chats"] ?? [])
            try check(records.first(where: { $0.id == savedID })?.turns.count == current?.turns.count, "Saved turns differ from the UI")
            try check(records.first(where: { $0.id == savedID })?.model == model, "Saved conversation lost its pinned model")
            passed("chat persisted on disk")
            #if os(iOS)
            try record("foreground replay")
            await sceneChanged(.inactive)
            try check(loadedModel == model && !busy, "Transient inactivity unloaded the model")
            await sceneChanged(.active)
            passed("transient inactivity preserves loaded model")
            await sceneChanged(.background)
            try await wait(60) { !busy && loadedModel == nil }
            await sceneChanged(.active)
            try await wait(300) { !busy }
            try check(loadedModel == model, "Model did not reload on foreground")
            draft = "What was our code word? Reply with only the word."
            await send(); try await wait(180) { !busy }
            let foreground = current?.turns.last?.assistant ?? ""
            report["foreground_reply"] = foreground
            try check(current?.turns.last?.complete == true && foreground.lowercased().contains("violet"), "Foreground replay failed: \(foreground)")
            passed("foreground reload and saved-history replay")
            try record("memory warning")
            await memoryWarning()
            await sceneChanged(.background)
            try await wait(60) { !busy && loadedModel == nil }
            await sceneChanged(.active)
            try check(!busy && loadedModel == nil, "Foreground transition reloaded after memory pressure")
            try check(!canSend, "Sending remained enabled after memory release")
            passed("memory warning releases model without automatic reload")
            #endif
            report["status"] = "passed"; report["chat_id"] = savedID
            try record("complete")
        } catch {
            report["status"] = "failed"; report["error"] = error.localizedDescription
            report["logs"] = Array(logs.suffix(30))
            try? record(report["stage"] as? String ?? "unknown")
            self.error = error.localizedDescription
        }
        return true
    }
    /// Isolate platform compiler compatibility without loading any weights.
    private func runMetalProbe() async {
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("metal-probe.json")
        var report: [String:Any] = ["status":"running", "platform":checkPlatform, "os":ProcessInfo.processInfo.operatingSystemVersionString]
        do {
            guard let device = MTLCreateSystemDefaultDevice() else { throw CocoaError(.featureUnsupported) }
            report["device"] = device.name
            report["apple2"] = device.supportsFamily(.apple2)
            report["apple9"] = device.supportsFamily(.apple9)
            report["apple10"] = device.supportsFamily(.apple10)
            var results = [[String:Any]]()
            for (type, explicitLanguage) in [("float", false), ("bfloat", false), ("bfloat", true)] {
                let options = MTLCompileOptions()
                if explicitLanguage { options.languageVersion = .version3_2 }
                options.mathMode = .safe
                options.mathFloatingPointFunctions = .precise
                let source = "#include <metal_stdlib>\nusing namespace metal; kernel void copy_value(device const \(type)* a [[buffer(0)]], device \(type)* b [[buffer(1)]], uint i [[thread_position_in_grid]]) { b[i] = a[i]; }"
                do {
                    let library = try await device.makeLibrary(source: source, options: options)
                    let pipeline = try await device.makeComputePipelineState(function: library.makeFunction(name: "copy_value")!)
                    let input = device.makeBuffer(length: 4, options: .storageModeShared)!
                    let output = device.makeBuffer(length: 4, options: .storageModeShared)!
                    let bits: UInt32 = type == "float" ? 0x3f800000 : 0x00003f80
                    input.contents().storeBytes(of: bits, as: UInt32.self)
                    output.contents().storeBytes(of: UInt32(0), as: UInt32.self)
                    let command = device.makeCommandQueue()!.makeCommandBuffer()!
                    let encoder = command.makeComputeCommandEncoder()!
                    encoder.setComputePipelineState(pipeline)
                    encoder.setBuffer(input, offset: 0, index: 0); encoder.setBuffer(output, offset: 0, index: 1)
                    encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                    encoder.endEncoding()
                    await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                        command.addCompletedHandler { _ in continuation.resume() }
                        command.commit()
                    }
                    let result = output.contents().load(as: UInt32.self)
                    results.append(["type":type,"explicit_msl32":explicitLanguage,"passed":command.status == .completed && result == bits, "bits":result])
                } catch { results.append(["type":type,"explicit_msl32":explicitLanguage,"passed":false,"error":error.localizedDescription]) }
            }
            report["results"] = results
            report["status"] = results.allSatisfy { $0["passed"] as? Bool == true } ? "passed" : "failed"
        } catch { report["status"] = "failed"; report["error"] = error.localizedDescription }
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
    }
}
#endif
