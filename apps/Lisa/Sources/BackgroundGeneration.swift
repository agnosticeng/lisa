import Foundation
#if os(iOS)
import BackgroundTasks
import UIKit
#endif

/// Owns one user-started generation's background GPU authorization.
@MainActor final class BackgroundGeneration {
    private var implementation: AnyObject?
    #if os(iOS)
    private var previousIdleTimer: Bool?
    #endif
    var onExpiration: (() -> Void)?
    var onLog: ((String) -> Void)?
    var isActive: Bool {
        #if os(iOS)
        if #available(iOS 26, *) { return (implementation as? ContinuedGeneration)?.isActive == true }
        #endif
        return false
    }
    func begin(image: Bool) async {
        #if os(iOS)
        previousIdleTimer = UIApplication.shared.isIdleTimerDisabled
        UIApplication.shared.isIdleTimerDisabled = true
        if #available(iOS 26, *) {
            if implementation == nil {
                implementation = ContinuedGeneration(
                    expired: { [weak self] in self?.onExpiration?() },
                    log: { [weak self] in self?.onLog?($0) })
            }
            await (implementation as? ContinuedGeneration)?.begin(image: image)
        }
        #endif
    }
    func cancelAdmission() {
        #if os(iOS)
        if #available(iOS 26, *), let session = implementation as? ContinuedGeneration, !session.isActive {
            session.finish(success: false)
        }
        #endif
    }
    func update(_ text: String, delta: Bool = false) {
        #if os(iOS)
        if #available(iOS 26, *) { (implementation as? ContinuedGeneration)?.update(text, delta: delta) }
        #endif
    }
    func finish(success: Bool) {
        #if os(iOS)
        if let previousIdleTimer {
            UIApplication.shared.isIdleTimerDisabled = previousIdleTimer
            self.previousIdleTimer = nil
        }
        if #available(iOS 26, *) { (implementation as? ContinuedGeneration)?.finish(success: success) }
        #endif
    }
}

#if os(iOS)
@available(iOS 26, *)
@MainActor private final class ContinuedGeneration {
    private let prefix = (Bundle.main.bundleIdentifier ?? "org.lisa.app") + ".generation."
    private var registered = false
    private var identifier: String?
    private var task: BGContinuedProcessingTask?
    private var admission: CheckedContinuation<Void, Never>?
    private var expired = false
    private var image = false
    private var tokens: Int64 = 0
    private let expiration: () -> Void
    private let log: (String) -> Void
    var isActive: Bool { task != nil && !expired }

    init(expired: @escaping () -> Void, log: @escaping (String) -> Void) {
        self.expiration = expired; self.log = log
    }
    func begin(image: Bool) async {
        guard identifier == nil else { return }
        guard BGTaskScheduler.supportedResources.contains(.gpu) else {
            log("Background GPU unavailable on this device; keep Lisa open while generating.")
            return
        }
        if !registered {
            registered = BGTaskScheduler.shared.register(forTaskWithIdentifier: prefix + "*", using: .main) { [weak self] incoming in
                Task { @MainActor in self?.accept(incoming) }
            }
        }
        guard registered else { log("Background generation registration failed"); return }
        self.image = image; expired = false; tokens = 0
        let id = prefix + UUID().uuidString
        identifier = id
        let request = BGContinuedProcessingTaskRequest(identifier: id,
            title: image ? "Creating an image" : "Writing a reply", subtitle: "Preparing…")
        request.requiredResources = .gpu
        request.strategy = .fail
        // Install the waiter before submission: the launch callback may arrive immediately.
        await withCheckedContinuation { continuation in
            admission = continuation
            Task {
                do {
                    if #available(iOS 27, *) {
                        try await BGTaskScheduler.shared.submitTaskRequest(request)
                    } else {
                        try BGTaskScheduler.shared.submit(request)
                    }
                } catch {
                    guard identifier == id else { return }
                    log("Background generation unavailable: \(error.localizedDescription). Keep Lisa open.")
                    finish(success: false)
                }
            }
            Task {
                try? await Task.sleep(for: .seconds(5))
                if identifier == id && task == nil {
                    log("Background generation admission timed out; keep Lisa open.")
                    finish(success: false)
                }
            }
        }
    }
    private func accept(_ incoming: BGTask) {
        guard let incoming = incoming as? BGContinuedProcessingTask,
              incoming.identifier == identifier else {
            incoming.setTaskCompleted(success: false); return
        }
        task = incoming
        incoming.progress.totalUnitCount = image ? 1000 : 512
        incoming.expirationHandler = { [weak self] in
            Task { @MainActor in
                guard let self, self.task?.identifier == incoming.identifier else { return }
                self.expired = true
                self.log("Background generation expired or was cancelled")
                self.expiration()
            }
        }
        log("Background GPU generation authorized")
        admission?.resume(); admission = nil
    }
    func update(_ text: String, delta: Bool) {
        guard let task, !expired else { return }
        if image {
            let pair = text.split(separator: " ").last?.split(separator: "/") ?? []
            guard pair.count == 2, let done = Int64(pair[0]), let total = Int64(pair[1]), total > 0 else { return }
            let value: Int64
            if text.hasPrefix("Encoding") { value = 0 }
            else if text.hasPrefix("Generating") { value = 100 + 650 * done / total }
            else if text.hasPrefix("Decoding") {
                // The initial stage notification uses 4/4, followed by actual tile counts.
                value = total > 4 ? 750 + 240 * done / total : 750
            } else { return }
            task.progress.completedUnitCount = max(task.progress.completedUnitCount, min(value, 990))
            task.updateTitle(task.title, subtitle: text)
        } else if delta {
            tokens += 1
            task.progress.totalUnitCount = max(512, tokens + 1)
            task.progress.completedUnitCount = tokens
            task.updateTitle(task.title, subtitle: "Writing your reply…")
        }
    }
    func finish(success: Bool) {
        if let task {
            if success { task.progress.completedUnitCount = task.progress.totalUnitCount }
            task.expirationHandler = nil
            task.setTaskCompleted(success: success && !expired)
        } else if let identifier {
            BGTaskScheduler.shared.cancel(taskRequestWithIdentifier: identifier)
        }
        task = nil; identifier = nil
        admission?.resume(); admission = nil
    }
}
#endif
