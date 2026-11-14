#if DEBUG && targetEnvironment(simulator)
import Foundation
import Metal

/// Opt-in integration check in the actual iOS app process, using its normal
/// AppStore, C bridge, engine thread, Metal runtime and persistent chat store.
/// No mock replies or remote inference are used. Results live in Documents.
@MainActor extension AppStore {
    func runSimulatorCheckIfRequested() async {
        if ProcessInfo.processInfo.arguments.contains("--lisa-metal-probe") { await runMetalProbe(); return }
        guard ProcessInfo.processInfo.arguments.contains("--lisa-simulator-check") else { return }
        let previousWeb = webEnabled, previousSpeech = speakReplies
        defer { webEnabled = previousWeb; speakReplies = previousSpeech }
        let model = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
        let resultURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("simulator-check.json")
        var report: [String:Any] = ["model":model, "platform":"iOS Simulator", "status":"running", "checks":[String](), "started":ISO8601DateFormatter().string(from: Date())]
        func record(_ stage: String) throws {
            report["stage"] = stage
            try FileManager.default.createDirectory(at: resultURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: resultURL, options: .atomic)
        }
        func check(_ condition: Bool, _ message: String) throws {
            if !condition { throw NSError(domain: "LisaSimulatorCheck", code: 1, userInfo: [NSLocalizedDescriptionKey:message]) }
        }
        func wait(_ timeout: TimeInterval, until condition: () -> Bool) async throws {
            let end = Date().addingTimeInterval(timeout)
            while !condition() {
                if let error { throw NSError(domain: "LisaSimulatorCheck", code: 2, userInfo: [NSLocalizedDescriptionKey:error]) }
                try check(Date() < end, "Timed out: \(report["stage"] ?? "unknown") — \(status)")
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
                try await wait(900) { downloading == nil && models.contains { $0.id == model && $0.local } }
            }
            passed("model downloaded")
            await newChat(); webEnabled = false; speakReplies = false
            var memory = try await bridge.call(["op":"models"]) as? [String:Any] ?? [:]
            memory.removeValue(forKey: "models"); report["memory"] = memory
            try record("load")
            await loadModel(model)
            try await wait(300) { !busy }
            try check(loadedModel == model, "Model did not load: \(error ?? status)")
            passed("Metal model loaded")
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
            passed("chat persisted on disk")
            report["status"] = "passed"; report["chat_id"] = savedID
            try record("complete")
        } catch {
            report["status"] = "failed"; report["error"] = error.localizedDescription
            try? record(report["stage"] as? String ?? "unknown")
            self.error = error.localizedDescription
        }
    }
    /// Isolate Simulator/host compiler compatibility without loading any weights.
    private func runMetalProbe() async {
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("metal-probe.json")
        var report: [String:Any] = ["status":"running", "os":ProcessInfo.processInfo.operatingSystemVersionString]
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
                    encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
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
