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
        <html><head><title>Search fixture</title></head><body><div id="rso">
        <div><article><a href="https://example.com/desktop"><h3>Desktop result</h3></a><p>Desktop evidence.</p><span>· Traduire cette page</span></article></div>
        <div><article style="content-visibility:hidden"><span>Mobile Site https://m.example.org</span><a href="https://www.google.com/goto?url=opaque-token&amp;gsessionid=session"><div><div role="heading" aria-level="3"><div role="link">Mobile result</div></div></div></a><p>Mobile result</p><p>Mobile evidence.</p></article></div>
        <div><article><a href="https://www.google.com/url?q=https%3A%2F%2Fexample.com%2Fredirect"><h3>Decoded redirect</h3></a></article></div>
        <div><article><a href="https://www.google.com/url?url=opaque"><h3>Opaque legacy redirect</h3></a><cite>https://www.cited.test › page</cite></article></div>
        <div><article><a href="https://www.google.com/url?q=CAESopaque&amp;sa=U"><h3>Opaque token</h3></a><span>tokens.test › docs</span><p>Token evidence.</p></article></div>
        <div><article><a href="https://example.com/desktop/news"><h3>Desktop result</h3></a><p>Site link.</p></article></div>
        </div>
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
        precondition(hits.map { $0["title"]! } == ["Desktop result", "Mobile result", "Decoded redirect", "Opaque legacy redirect", "Opaque token"])
        precondition(hits[0]["url"] == "https://example.com/desktop")
        precondition(hits[1]["url"] == "https://www.google.com/goto?url=opaque-token&gsessionid=session")
        precondition(hits[2]["url"] == "https://example.com/redirect")
        precondition(hits[3]["url"] == "https://www.google.com/url?url=opaque")
        precondition(hits[4]["url"] == "https://www.google.com/url?q=CAESopaque&sa=U")
        precondition(hits.map { $0["source"]! } == ["example.com", "m.example.org", "example.com", "cited.test", "tokens.test"])
        precondition(hits[0]["snippet"] == "Desktop evidence.")
        precondition(hits[1]["snippet"] == "Mobile evidence.")
        precondition(hits[3]["snippet"] == "")
        precondition(hits[4]["snippet"] == "tokens.test › docs Token evidence.")
        print("Browser extraction passed: desktop/mobile cards, sources, redirects, duplicate, site-link and non-result filtering")
    }
}

@main struct RunBrowserScriptCheck {
    @MainActor static func main() async throws {
        _ = NSApplication.shared
        try await BrowserScriptCheck().run()
    }
}
