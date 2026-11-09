use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use block2::RcBlock;
use objc2::MainThreadOnly;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{DefinedClass, MainThreadMarker, define_class, msg_send};
use objc2_app_kit::{NSBackingStoreType, NSPanel, NSWindowCollectionBehavior, NSWindowStyleMask};
use objc2_foundation::{
    NSError, NSObject, NSObjectProtocol, NSRect, NSSize, NSString, NSURL, NSURLRequest,
};
use objc2_web_kit::{WKNavigation, WKNavigationDelegate, WKWebView, WKWebViewConfiguration};
use serde::Deserialize;

const SEARCH_TIMEOUT: Duration = Duration::from_secs(25);
const USER_INPUT_TIMEOUT: Duration = Duration::from_secs(300);
const SEARCH_GRACE: Duration = Duration::from_millis(350);
const SEARCH_RETRY_INTERVAL: Duration = Duration::from_secs(1);
const RESULT_LIMIT: usize = 8;

#[derive(Deserialize)]
struct SearchPage {
    title: String,
    url: String,
    results: Vec<SearchHit>,
    body: String,
}

#[derive(Deserialize)]
struct PageContent {
    title: String,
    url: String,
    body: String,
    links: Vec<PageLink>,
}

#[derive(Deserialize)]
struct PageLink {
    title: String,
    url: String,
}

#[derive(Deserialize)]
struct SearchHit {
    title: String,
    url: String,
    snippet: String,
}

#[derive(Debug, Default)]
struct NavigationState {
    current: RefCell<Option<Retained<WKNavigation>>>,
    ready: Cell<bool>,
    error: RefCell<Option<String>>,
}

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[ivars = NavigationState]
    #[derive(Debug)]
    struct NavigationObserver;

    unsafe impl NSObjectProtocol for NavigationObserver {}
    unsafe impl WKNavigationDelegate for NavigationObserver {
        #[unsafe(method(webView:didStartProvisionalNavigation:))]
        unsafe fn started(&self, _view: &WKWebView, navigation: Option<&WKNavigation>) {
            *self.ivars().current.borrow_mut() = navigation.and_then(|nav| unsafe {
                Retained::retain(nav as *const WKNavigation as *mut WKNavigation)
            });
            self.ivars().ready.set(false);
            self.ivars().error.borrow_mut().take();
        }

        #[unsafe(method(webView:didFinishNavigation:))]
        unsafe fn completed(&self, _view: &WKWebView, navigation: Option<&WKNavigation>) {
            if self.is_current(navigation) {
                self.ivars().ready.set(true);
            }
        }

        #[unsafe(method(webView:didFailProvisionalNavigation:withError:))]
        unsafe fn failed_provisional(
            &self,
            _view: &WKWebView,
            navigation: Option<&WKNavigation>,
            error: &NSError,
        ) {
            self.record_error(navigation, error);
        }

        #[unsafe(method(webView:didFailNavigation:withError:))]
        unsafe fn failed(
            &self,
            _view: &WKWebView,
            navigation: Option<&WKNavigation>,
            error: &NSError,
        ) {
            self.record_error(navigation, error);
        }
    }
);

impl NavigationObserver {
    fn is_current(&self, navigation: Option<&WKNavigation>) -> bool {
        let current = self.ivars().current.borrow();
        match (current.as_deref(), navigation) {
            (Some(current), Some(navigation)) => std::ptr::eq(current, navigation),
            _ => false,
        }
    }

    fn record_error(&self, navigation: Option<&WKNavigation>, error: &NSError) {
        if self.is_current(navigation) {
            self.ivars().ready.set(false);
            *self.ivars().error.borrow_mut() = Some(error.localizedDescription().to_string());
        }
    }
}

/// An app-owned browser panel used by the UI agent to search and read pages.
/// Content extraction runs inside the loaded page; no search API key or
/// separate search endpoint is involved.
#[derive(Debug)]
pub(crate) struct WebSearchBrowser {
    panel: Retained<NSPanel>,
    web_view: Retained<WKWebView>,
    observer: Retained<NavigationObserver>,
    pending: Option<PendingNavigation>,
}

#[derive(Debug)]
struct PendingNavigation {
    reply: Sender<BrowserEvent>,
    started: Instant,
    last_extract: Option<Instant>,
    extracting: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
    prompt_sent: Arc<AtomicBool>,
    kind: NavigationKind,
}

#[derive(Debug, Clone, Copy)]
enum NavigationKind {
    Search,
    Page,
}

#[derive(Debug)]
pub(crate) enum BrowserEvent {
    NeedsInput(String),
    Finished(Result<String, String>),
}

#[derive(Debug, Clone)]
pub(crate) enum BrowserRequest {
    Search(String),
    Open(String),
}

impl WebSearchBrowser {
    pub(crate) fn new(mtm: MainThreadMarker) -> Self {
        let panel = NSPanel::initWithContentRect_styleMask_backing_defer(
            NSPanel::alloc(mtm),
            NSRect::new(
                objc2_foundation::NSPoint::new(0.0, 0.0),
                NSSize::new(820.0, 620.0),
            ),
            NSWindowStyleMask::Titled | NSWindowStyleMask::Closable | NSWindowStyleMask::Resizable,
            NSBackingStoreType::Buffered,
            false,
        );
        unsafe { panel.setReleasedWhenClosed(false) };
        panel.setTitle(&NSString::from_str("Lisa Browser"));
        panel.setCollectionBehavior(
            NSWindowCollectionBehavior::FullScreenAuxiliary
                | NSWindowCollectionBehavior::MoveToActiveSpace,
        );
        let config = unsafe { WKWebViewConfiguration::new(mtm) };
        let web_view = unsafe {
            WKWebView::initWithFrame_configuration(
                WKWebView::alloc(mtm),
                NSRect::new(
                    objc2_foundation::NSPoint::new(0.0, 0.0),
                    NSSize::new(820.0, 620.0),
                ),
                &config,
            )
        };
        let observer = NavigationObserver::alloc(mtm).set_ivars(NavigationState::default());
        let observer: Retained<NavigationObserver> = unsafe { msg_send![super(observer), init] };
        unsafe { web_view.setNavigationDelegate(Some(ProtocolObject::from_ref(&*observer))) };
        panel.setContentView(Some(&web_view));
        Self {
            panel,
            web_view,
            observer,
            pending: None,
        }
    }

    pub(crate) fn search(
        &mut self,
        query: &str,
        reply: Sender<BrowserEvent>,
    ) -> anyhow::Result<()> {
        if self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.finished.load(Ordering::Acquire))
        {
            self.pending = None;
        }
        anyhow::ensure!(self.pending.is_none(), "a web search is already running");
        anyhow::ensure!(!query.trim().is_empty(), "web search query is empty");
        anyhow::ensure!(query.len() <= 1000, "web search query is too long");
        self.navigate(&google_search_url(query), NavigationKind::Search, reply)
    }

    pub(crate) fn open(&mut self, url: &str, reply: Sender<BrowserEvent>) -> anyhow::Result<()> {
        let url = url.trim();
        anyhow::ensure!(url.len() <= 2048, "web page URL is too long");
        anyhow::ensure!(
            url.starts_with("https://") || url.starts_with("http://"),
            "web page URL must use HTTP or HTTPS"
        );
        self.navigate(url, NavigationKind::Page, reply)
    }

    fn navigate(
        &mut self,
        target: &str,
        kind: NavigationKind,
        reply: Sender<BrowserEvent>,
    ) -> anyhow::Result<()> {
        if self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.finished.load(Ordering::Acquire))
        {
            self.pending = None;
        }
        anyhow::ensure!(
            self.pending.is_none(),
            "a browser navigation is already running"
        );
        let url = NSURL::URLWithString(&NSString::from_str(target))
            .ok_or_else(|| anyhow::anyhow!("could not construct browser URL"))?;
        anyhow::ensure!(
            url.host().is_some_and(|host| !host.is_empty()),
            "web URL must have a host"
        );
        let request = NSURLRequest::requestWithURL(&url);
        self.observer.ivars().ready.set(false);
        self.observer.ivars().error.borrow_mut().take();
        self.pending = Some(PendingNavigation {
            reply,
            started: Instant::now(),
            last_extract: None,
            extracting: Arc::new(AtomicBool::new(false)),
            finished: Arc::new(AtomicBool::new(false)),
            prompt_sent: Arc::new(AtomicBool::new(false)),
            kind,
        });
        if unsafe { self.web_view.loadRequest(&request) }.is_none() {
            self.pending = None;
            anyhow::bail!("browser navigation could not be started");
        }
        // Keep the browser visible so the user can follow the model's research.
        self.panel.orderFrontRegardless();
        Ok(())
    }

    /// Called from the UI's existing poll timer. Wait for WebKit navigation to
    /// settle, then serialize a small set of result cards from the rendered DOM.
    pub(crate) fn poll(&mut self) {
        let Some(pending) = self.pending.as_ref() else {
            return;
        };
        if pending.finished.load(Ordering::Acquire) {
            self.pending = None;
            return;
        }
        let navigation_error = self.observer.ivars().error.borrow_mut().take();
        if let Some(error) = navigation_error {
            let pending = self.pending.take().expect("pending navigation");
            pending.finished.store(true, Ordering::Release);
            let _ = pending.reply.send(BrowserEvent::Finished(Err(format!(
                "could not load page: {error}"
            ))));
            return;
        }
        let timeout = if pending.prompt_sent.load(Ordering::Acquire) {
            USER_INPUT_TIMEOUT
        } else {
            SEARCH_TIMEOUT
        };
        if pending.started.elapsed() >= timeout {
            let pending = self.pending.take().expect("pending navigation");
            pending.finished.store(true, Ordering::Release);
            unsafe { self.web_view.stopLoading() };
            let message = match pending.kind {
                NavigationKind::Search => {
                    "Google search timed out. Check the browser panel for a consent or verification prompt, then retry."
                }
                NavigationKind::Page => "browser page navigation timed out",
            };
            let _ = pending
                .reply
                .send(BrowserEvent::Finished(Err(message.into())));
            return;
        }
        if pending.extracting.load(Ordering::Acquire)
            || pending.started.elapsed() < SEARCH_GRACE
            || pending
                .last_extract
                .is_some_and(|last| last.elapsed() < SEARCH_RETRY_INTERVAL)
            || !self.observer.ivars().ready.get()
            || unsafe { self.web_view.isLoading() }
        {
            return;
        }

        let pending = self.pending.as_mut().expect("pending search");
        pending.last_extract = Some(Instant::now());
        pending.extracting.store(true, Ordering::Release);
        let reply = pending.reply.clone();
        let extracting = pending.extracting.clone();
        let finished = pending.finished.clone();
        let prompt_sent = pending.prompt_sent.clone();
        let kind = pending.kind;
        let extraction = match kind {
            NavigationKind::Search => SEARCH_SCRIPT,
            NavigationKind::Page => PAGE_SCRIPT,
        };
        let script = NSString::from_str(&format!(
            "(() => {{ {GOOGLE_PROMPT_SCRIPT} return {extraction}; }})()"
        ));
        let block = RcBlock::new(move |result: *mut AnyObject, error: *mut NSError| {
            if finished.load(Ordering::Acquire) {
                return;
            }
            let json = if error.is_null() && !result.is_null() {
                Some(unsafe { &*(result as *mut NSString) }.to_string())
            } else {
                None
            };
            let needs_input = json
                .as_deref()
                .and_then(|json| serde_json::from_str::<serde_json::Value>(json).ok())
                .and_then(|value| {
                    value
                        .get("needs_input")
                        .and_then(|value| value.as_str())
                        .map(str::to_string)
                });
            if let Some(message) = needs_input {
                if !prompt_sent.swap(true, Ordering::AcqRel) {
                    let _ = reply.send(BrowserEvent::NeedsInput(message));
                }
                extracting.store(false, Ordering::Release);
                return;
            }
            let response = if !error.is_null() {
                let error = unsafe { &*error };
                Err(format!(
                    "browser page error: {}",
                    error.localizedDescription()
                ))
            } else if result.is_null() {
                Err("browser page returned no data".into())
            } else {
                let json = json.expect("non-null JavaScript result has JSON");
                match kind {
                    NavigationKind::Search => parse_search_page(&json),
                    NavigationKind::Page => parse_page(&json),
                }
            };
            if matches!(kind, NavigationKind::Search)
                && response.as_ref().is_err_and(|error| {
                    error.starts_with("Google returned no readable result cards")
                })
            {
                // Google may still be showing a consent/challenge page. Keep
                // the request alive so a user can resolve it in the visible
                // panel; poll() retries extraction after the DOM changes.
                extracting.store(false, Ordering::Release);
                return;
            }
            finished.store(true, Ordering::Release);
            let _ = reply.send(BrowserEvent::Finished(response));
            extracting.store(false, Ordering::Release);
        });
        unsafe {
            self.web_view
                .evaluateJavaScript_completionHandler(&script, Some(&block));
        }
    }
}

// Detect Google interstitials without making a consent or verification choice.
// The UI asks the user to resolve them in the visible browser, then polling
// resumes extraction after the interstitial is gone.
const GOOGLE_PROMPT_SCRIPT: &str = r#"
  if (location.hostname === 'consent.google.com') {
    return JSON.stringify({needs_input:
      "Google needs your consent before browsing can continue. Please review the prompt in Lisa Browser and choose whether to accept or reject it. I’ll wait for your choice."});
  }
  const googleHost = location.hostname === 'google.com' || location.hostname.endsWith('.google.com');
  if (googleHost && (location.pathname.startsWith('/sorry') ||
      Array.from(document.querySelectorAll('iframe[src*="recaptcha"]')).some(frame => frame.getClientRects().length > 0))) {
    return JSON.stringify({needs_input:
      "Google is asking for verification. Please complete the prompt yourself in Lisa Browser. I’ll continue once it is resolved."});
  }
"#;

const SEARCH_SCRIPT: &str = r#"(() => {
  const clean = value => (value || '').replace(/\s+/g, ' ').trim();
  const hosts = new Set(['google.com', 'www.google.com', 'consent.google.com']);
  const results = Array.from(document.querySelectorAll('a'))
    .map(a => {
      const heading = a.querySelector('h3');
      if (!heading) return null;
      let url = a.href;
      try {
        let parsed = new URL(url);
        if (hosts.has(parsed.hostname)) {
          const target = parsed.searchParams.get('q') || parsed.searchParams.get('url') || '';
          if (/^https?:\/\//i.test(target)) {
            url = target;
            parsed = new URL(url);
          } else if (parsed.pathname !== '/url') {
            return null;
          }
        }
        if (!['http:', 'https:'].includes(parsed.protocol)) return null;
        if (hosts.has(parsed.hostname) && parsed.pathname !== '/url') return null;
      } catch (_) { return null; }
      let card = a;
      for (let i = 0; i < 5 && card.parentElement; i++) {
        card = card.parentElement;
        const text = clean(card.innerText);
        if (text.length > clean(heading.innerText).length + 20 && text.length < 1400) break;
      }
      const title = clean(heading.innerText);
      const text = clean(card.innerText);
      const snippet = text.startsWith(title) ? text.slice(title.length).trim() : text;
      return {title, url, snippet: snippet.slice(0, 700)};
    })
    .filter(Boolean)
    .filter((item, index, all) => all.findIndex(other => other.url === item.url) === index)
    .slice(0, 8);
  return JSON.stringify({title: clean(document.title), url: location.href,
    results, body: clean(document.body ? document.body.innerText : '').slice(0, 1800)});
})()"#;

const PAGE_SCRIPT: &str = r#"(() => {
  const clean = value => (value || '').replace(/\s+/g, ' ').trim();
  const root = document.querySelector('main, article, [role="main"]') || document.body;
  const links = Array.from(root ? root.querySelectorAll('a[href]') : [])
    .map(a => ({title: clean(a.innerText || a.getAttribute('aria-label') || ''), url: a.href}))
    .filter(link => {
      try { return link.title.length > 0 && /^https?:$/.test(new URL(link.url).protocol); }
      catch (_) { return false; }
    })
    .filter((link, index, all) => all.findIndex(other => other.url === link.url) === index)
    .slice(0, 24);
  return JSON.stringify({title: clean(document.title), url: location.href,
    body: (root ? root.innerText : '').trim().slice(0, 12000), links});
})()"#;

fn google_search_url(query: &str) -> String {
    let mut encoded = String::with_capacity(query.len());
    for byte in query.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char)
            }
            b' ' => encoded.push('+'),
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    format!("https://www.google.com/search?q={encoded}&num={RESULT_LIMIT}")
}

fn parse_search_page(json: &str) -> Result<String, String> {
    let page: SearchPage = serde_json::from_str(json)
        .map_err(|error| format!("could not read Google search results: {error}"))?;
    if page.results.is_empty() {
        return Err(format!(
            "Google returned no readable result cards (page title: {}). It may have shown a consent or challenge page.",
            page.title
        ));
    }

    let mut out = format!("Search page: {}\n", page.url);
    for (index, result) in page.results.iter().take(RESULT_LIMIT).enumerate() {
        let title = result.title.trim();
        let url = result.url.trim();
        if !(url.starts_with("https://") || url.starts_with("http://")) || title.is_empty() {
            continue;
        }
        out.push_str(&format!(
            "\n{}. {}\nURL: {}\nSnippet: {}\n",
            index + 1,
            title,
            url,
            result.snippet.trim()
        ));
    }
    if out.lines().count() < 4 {
        Err(format!(
            "Google's search page did not expose parseable results. Page text: {}",
            page.body
        ))
    } else {
        Ok(out)
    }
}

fn parse_page(json: &str) -> Result<String, String> {
    let page: PageContent = serde_json::from_str(json)
        .map_err(|error| format!("could not extract readable page content: {error}"))?;
    let url = page.url.trim();
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        return Err("the browser redirected to a non-web URL".into());
    }
    let body = page.body.trim();
    if body.is_empty() {
        return Err(format!("{} contained no readable page text", url));
    }
    let mut out = format!("Page: {}\nURL: {}\n\n{}", page.title.trim(), url, body);
    let links = page
        .links
        .iter()
        .filter(|link| link.url.starts_with("https://") || link.url.starts_with("http://"))
        .take(24)
        .collect::<Vec<_>>();
    if !links.is_empty() {
        out.push_str("\n\nLinks on this page:");
        for link in links {
            out.push_str(&format!("\n- {} — {}", link.title.trim(), link.url.trim()));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn search_encoding_preserves_query_not_url_syntax() {
        assert_eq!(
            google_search_url("Rust & café?#"),
            "https://www.google.com/search?q=Rust+%26+caf%C3%A9%3F%23&num=8"
        );
    }

    #[test]
    fn extraction_rejects_empty_or_non_web_pages() {
        assert!(parse_search_page(&json!({"title":"Consent", "url":"https://consent.google.com", "body":"Accept all", "results":[]}).to_string()).unwrap_err().contains("no readable result cards"));
        for (url, body) in [("file:///tmp/test", "text"), ("https://example.com", "   ")] {
            assert!(
                parse_page(
                    &json!({"title":"Page", "url":url, "body":body, "links":[]}).to_string()
                )
                .is_err()
            );
        }
        let result = parse_page(&json!({"title":"Page", "url":"https://example.com", "body":"First paragraph\n\nSecond paragraph", "links":[
            {"title":"Next", "url":"https://example.com/next"},
            {"title":"Script", "url":"javascript:alert(1)"}
        ]}).to_string()).unwrap();
        assert!(result.contains("First paragraph\n\nSecond paragraph"));
        assert!(result.contains("https://example.com/next"));
        assert!(!result.contains("javascript:"));
    }
}
