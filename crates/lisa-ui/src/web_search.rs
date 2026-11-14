use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use crate::browser::*;
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

const SEARCH_TIMEOUT: Duration = Duration::from_secs(25);
const USER_INPUT_TIMEOUT: Duration = Duration::from_secs(300);
const SEARCH_GRACE: Duration = Duration::from_millis(350);
const SEARCH_RETRY_INTERVAL: Duration = Duration::from_secs(1);

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
