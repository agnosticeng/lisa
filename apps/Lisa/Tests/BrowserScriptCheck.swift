import AppKit
import WebKit

/// Executes the production extractor in WebKit against desktop and mobile cards.
@MainActor final class BrowserScriptCheck: NSObject, WKNavigationDelegate {
    let view = WKWebView()
    var loaded: CheckedContinuation<Void, Error>?

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        loaded?.resume(); loaded = nil
    }
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        loaded?.resume(throwing: error); loaded = nil
    }
    func run() async throws {
        let source = try String(contentsOfFile: "crates/lisa-app/src/browser.rs", encoding: .utf8)
        let script = source.components(separatedBy: "const SEARCH_SCRIPT: &str = r#\"")[1]
            .components(separatedBy: "\"#;")[0]
        view.navigationDelegate = self
        let fixture = """
        <html><head><title>Search fixture</title></head><body>
        <article><a href="https://example.com/desktop"><h3>Desktop result</h3></a><p>Desktop evidence.</p></article>
        <article style="content-visibility:hidden"><a href="https://www.google.com/goto?url=opaque-token&amp;gsessionid=session"><div><div role="heading" aria-level="3"><div role="link">Mobile result</div></div></div></a><p>Mobile evidence.</p></article>
        <article><a href="https://www.google.com/url?q=https%3A%2F%2Fexample.com%2Fredirect"><h3>Decoded redirect</h3></a></article>
        <article><a href="https://www.google.com/url?url=opaque"><h3>Opaque legacy redirect</h3></a></article>
        <a href="https://example.com/desktop"><h3>Duplicate</h3></a>
        <a href="https://www.google.com/search?q=other"><h3>Search navigation</h3></a>
        <a href="javascript:alert(1)"><h3>Not a web link</h3></a>
        <a href="https://example.com/no-heading">Not a result</a>
        </body></html>
        """
        try await withCheckedThrowingContinuation { continuation in
            loaded = continuation
            view.loadHTMLString(fixture, baseURL: URL(string: "https://www.google.com/"))
        }
        let json = try await view.evaluateJavaScript(script) as! String
        let result = try JSONSerialization.jsonObject(with: Data(json.utf8)) as! [String: Any]
        let hits = result["results"] as! [[String: String]]
        precondition(hits.map { $0["title"]! } == ["Desktop result", "Mobile result", "Decoded redirect", "Opaque legacy redirect"])
        precondition(hits[0]["url"] == "https://example.com/desktop")
        precondition(hits[1]["url"] == "https://www.google.com/goto?url=opaque-token&gsessionid=session")
        precondition(hits[2]["url"] == "https://example.com/redirect")
        precondition(hits[3]["url"] == "https://www.google.com/url?url=opaque")
        precondition(hits[0]["snippet"]?.contains("Desktop evidence.") == true)
        precondition(hits[1]["snippet"]?.contains("Mobile evidence.") == true)
        print("Browser extraction passed: desktop/mobile headings, redirects, duplicate and non-result filtering")
    }
}

@main struct RunBrowserScriptCheck {
    @MainActor static func main() async throws {
        _ = NSApplication.shared
        try await BrowserScriptCheck().run()
    }
}
