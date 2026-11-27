import SwiftUI
import WebKit

@MainActor final class BrowserController: NSObject, ObservableObject, WKNavigationDelegate {
    // A nonzero viewport keeps offscreen WebKit result cards laid out. Without
    // Safari's product token Google serves a reduced page with weaker results.
    let webView: WKWebView = {
        let configuration = WKWebViewConfiguration()
        configuration.applicationNameForUserAgent = BrowserController.safariProduct
        return WKWebView(frame: CGRect(x: 0, y: 0, width: 390, height: 844), configuration: configuration)
    }()
    /// Safari's version tracks the OS major version from release 26 onward.
    static var safariProduct: String {
        let version = "Version/\(max(ProcessInfo.processInfo.operatingSystemVersion.majorVersion, 26)).0"
        #if os(iOS)
        return "\(version) Mobile/15E148 Safari/604.1"
        #else
        return "\(version) Safari/605.1.15"
        #endif
    }
    @Published var address = ""
    @Published var status = ""
    @Published private(set) var needsInput = false
    private var navigation: WKNavigation?
    private var ready = false
    private var generation = UUID()
    private var extraction: Task<Void, Never>?
    private var completion: (([String:Any]) async -> Void)?
    private var requestID: Any?
    override init() { super.init(); webView.navigationDelegate = self }
    func navigate(_ request: [String:Any], completion: @escaping ([String:Any]) async -> Void) {
        cancel()
        generation = UUID(); let token = generation
        self.completion = completion; requestID = request["id"]
        guard let raw = request["url"] as? String, let url = URL(string: raw), ["http", "https"].contains(url.scheme?.lowercased() ?? ""), let script = request["script"] as? String else {
            finish(["error":"The browser only accepts HTTP and HTTPS URLs"]); return
        }
        address = raw; status = "Loading…"; ready = false
        navigation = webView.load(URLRequest(url: url))
        let search = request["search"] as? Bool ?? false
        extraction = Task { [weak self] in
            let started = Date(); var prompted = false
            while !Task.isCancelled {
                guard let self, self.generation == token else { return }
                let timeout: TimeInterval = prompted ? 300 : 25
                if Date().timeIntervalSince(started) > timeout { self.finish(["error":"Browser navigation timed out"]); return }
                if self.ready && !self.webView.isLoading {
                    do {
                        let result = try await self.webView.evaluateJavaScript(script)
                        guard !Task.isCancelled, self.generation == token else { return }
                        guard let json = result as? String, let bytes = json.data(using: .utf8), let page = try JSONSerialization.jsonObject(with: bytes) as? [String:Any] else {
                            self.finish(["error":"The page returned no readable content"]); return
                        }
                        if let message = page["needs_input"] as? String {
                            self.status = message; self.needsInput = true
                            if !prompted {
                                prompted = true
                                await completion(["op":"browser_result", "id":request["id"] ?? NSNull(), "needs_input":message])
                            }
                        } else if search && (page["results"] as? [Any] ?? []).isEmpty {
                            self.status = "Waiting for readable search results…"
                        } else {
                            self.status = "Page read"; self.address = self.webView.url?.absoluteString ?? raw
                            self.finish(["json":json, "search":search]); return
                        }
                    } catch {
                        if !Task.isCancelled, self.generation == token { self.finish(["error":error.localizedDescription]) }
                        return
                    }
                }
                // Read a finished page promptly; a pending consent or
                // verification prompt waits on the user, so poll it slowly.
                try? await Task.sleep(for: .milliseconds(prompted ? 1000 : 200))
            }
        }
    }
    private func finish(_ result: [String:Any]) {
        guard let completion, let id = requestID else { return }
        needsInput = false
        self.completion = nil; requestID = nil; extraction?.cancel(); extraction = nil
        var response = result; response["op"] = "browser_result"; response["id"] = id
        Task { await completion(response) }
    }
    func cancel() {
        generation = UUID(); webView.stopLoading(); extraction?.cancel(); extraction = nil
        finish(["error":"Browser request cancelled"])
    }
    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) { self.navigation = navigation; ready = false }
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { if self.navigation === navigation { ready = true } }
    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        if self.navigation === navigation { finish(["error":error.localizedDescription]) }
    }
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        if self.navigation === navigation { finish(["error":error.localizedDescription]) }
    }
    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        let scheme = navigationAction.request.url?.scheme?.lowercased() ?? ""
        decisionHandler(["http", "https", "about"].contains(scheme) ? .allow : .cancel)
    }
}

struct BrowserView: View {
    @ObservedObject var controller: BrowserController
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(controller.address).font(.caption).lineLimit(1).textSelection(.enabled)
                    Text(controller.status).font(.caption).foregroundStyle(.secondary)
                }.frame(maxWidth: .infinity, alignment: .leading).padding()
                Divider()
                BrowserSurface(webView: controller.webView)
            }
            .navigationTitle("Lisa Browser")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }.frame(idealWidth: 900, idealHeight: 700)
    }
}
#if os(macOS)
struct BrowserSurface: NSViewRepresentable {
    let webView: WKWebView
    func makeNSView(context: Context) -> WKWebView { webView }
    func updateNSView(_ view: WKWebView, context: Context) {}
}
#else
struct BrowserSurface: UIViewRepresentable {
    let webView: WKWebView
    func makeUIView(context: Context) -> WKWebView { webView }
    func updateUIView(_ view: WKWebView, context: Context) {}
}
#endif
