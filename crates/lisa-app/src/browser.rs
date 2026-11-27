//! Messages between the engine worker and the platform browser.
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

/// Navigation is limited to links returned by the latest successful Google
/// search. Numeric references avoid asking small models to copy opaque URLs.
/// One refined search is allowed; after that a result must be opened, since
/// each further search costs a navigation and a model pass without adding a
/// source.
#[derive(Default)]
pub(crate) struct SearchNavigation {
    results: Vec<String>,
    unread_searches: usize,
}

const SEARCHES_BEFORE_READING: usize = 2;

impl SearchNavigation {
    pub(crate) fn navigate(
        &mut self,
        request: BrowserRequest,
        navigate: impl FnOnce(BrowserRequest) -> String,
    ) -> String {
        match request {
            BrowserRequest::Search(_) if self.unread_searches >= SEARCHES_BEFORE_READING => {
                "Browser error: open the most relevant result from the latest search with web_open before searching again.".into()
            }
            BrowserRequest::Search(query) => {
                let content = navigate(BrowserRequest::Search(query));
                if content.starts_with("Search page: ") {
                    self.unread_searches += 1;
                    self.results = content
                        .lines()
                        .filter_map(|line| line.strip_prefix("URL: ").map(str::to_string))
                        .collect();
                    // Keep full redirects in the navigator, not the prompt.
                    // Their opaque tokens waste context and are easy to miscopy.
                    return content
                        .lines()
                        .filter(|line| !line.starts_with("URL: "))
                        .collect::<Vec<_>>()
                        .join("\n");
                }
                content
            }
            BrowserRequest::Open(reference) => {
                let url = if let Some(index) = reference.strip_prefix("google-result:") {
                    index
                        .parse::<usize>()
                        .ok()
                        .and_then(|n| n.checked_sub(1))
                        .and_then(|index| self.results.get(index))
                } else {
                    self.results.iter().find(|url| **url == reference)
                };
                match url {
                    Some(url) => {
                        self.unread_searches = 0;
                        navigate(BrowserRequest::Open(url.clone()))
                    }
                    None => "Browser error: choose a result number from the latest Google search. Only Google result links can be opened; do not invent a URL.".into(),
                }
            }
        }
    }
}

use serde::Deserialize;
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
    #[serde(default)]
    source: String,
    snippet: String,
}

// Detect Google interstitials without making a consent or verification choice.
// The UI asks the user to resolve them in the visible browser, then polling
// resumes extraction after the interstitial is gone.
pub(crate) const GOOGLE_PROMPT_SCRIPT: &str = r#"
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

pub(crate) const SEARCH_SCRIPT: &str = r#"(() => {
  const clean = value => (value || '').replace(/\s+/g, ' ').trim();
  // Hidden cards have no innerText; join their text nodes so adjacent
  // elements do not run together.
  const words = node => {
    const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
    const parts = [];
    while (walker.nextNode()) parts.push(walker.currentNode.nodeValue);
    return parts.join(' ');
  };
  const text = node => clean(node.innerText) || clean(words(node));
  const hosts = new Set(['google.com', 'www.google.com', 'consent.google.com']);
  // Result links may be opaque /url or /goto redirects; WebKit resolves them on open.
  const redirects = new Set(['/url', '/goto']);
  const headings = 'h3, [role="heading"][aria-level="3"]';
  const translate = /\s*·?\s*(?:Traduire cette page|Translate this page|Diese Seite übersetzen|Traducir esta página|Traduci questa pagina)\s*/gi;
  const domain = value => value.replace(/^https?:\/\//i, '').split(/[\s›\/]/)[0].replace(/^www\./, '');
  const results = Array.from(document.querySelectorAll('a'))
    .map(a => {
      const heading = a.querySelector(headings);
      if (!heading) return null;
      let url = a.href, source = '';
      try {
        let parsed = new URL(url);
        if (hosts.has(parsed.hostname)) {
          const target = parsed.searchParams.get('q') || parsed.searchParams.get('url') || '';
          if (/^https?:\/\//i.test(target)) {
            url = target;
            parsed = new URL(url);
          } else if (!redirects.has(parsed.pathname)) {
            return null;
          }
        }
        if (!['http:', 'https:'].includes(parsed.protocol)) return null;
        if (hosts.has(parsed.hostname) && !redirects.has(parsed.pathname)) return null;
        if (!hosts.has(parsed.hostname)) source = domain(parsed.hostname);
      } catch (_) { return null; }
      // The card is the widest ancestor that still holds only this result.
      let card = a;
      while (card.parentElement && card.parentElement !== document.body) {
        const parent = card.parentElement;
        if (parent.querySelectorAll(headings).length > 1 || clean(parent.textContent).length > 1400) break;
        card = parent;
      }
      const title = text(heading);
      if (!title) return null;
      const body = clean(text(card).split(title).join(' '));
      const cite = card.querySelector('cite');
      const shown = cite ? text(cite) : '';
      const address = body.match(/https?:\/\/[^\s›]+/i);
      if (!source) {
        // Opaque redirects hide the destination; cards still print its address.
        source = domain(shown || (address || body.match(/(?:[a-z0-9-]+\.)+[a-z]{2,}(?= ›)/i) || [''])[0]);
      }
      // The section label, site name and address before the description
      // repeat the source; the translation offer is a Google control.
      const end = shown && body.includes(shown) ? body.indexOf(shown) + shown.length
        : address ? address.index + address[0].length : 0;
      const snippet = clean(body.slice(end).replace(translate, ' ')).replace(/^[·—\s]+/, '');
      return {title, url, source, snippet: snippet.slice(0, 700)};
    })
    .filter(Boolean)
    // Site links repeat one result's title; keep the first card of each.
    .filter((item, index, all) => all.findIndex(other => other.url === item.url ||
      (other.title === item.title && other.source === item.source)) === index)
    .slice(0, 8);
  return JSON.stringify({title: clean(document.title), url: location.href,
    results, body: clean(document.body ? document.body.innerText : '').slice(0, 1800)});
})()"#;

pub(crate) const PAGE_SCRIPT: &str = r#"(() => {
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

pub(crate) fn google_search_url(query: &str) -> String {
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

pub(crate) fn parse_search_page(json: &str) -> Result<String, String> {
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
        out.push_str(&format!("\n{}. {}\nURL: {}\n", index + 1, title, url));
        let source = result.source.trim();
        if !source.is_empty() {
            out.push_str(&format!("Source: {source}\n"));
        }
        out.push_str(&format!("Snippet: {}\n", result.snippet.trim()));
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

pub(crate) fn parse_page(json: &str) -> Result<String, String> {
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
    fn navigation_only_opens_results_from_the_latest_google_search() {
        let mut navigation = SearchNavigation::default();
        let blocked = navigation.navigate(
            BrowserRequest::Open("https://www.cherche.com/invented".into()),
            |_| panic!("invented URL opened"),
        );
        assert!(blocked.starts_with("Browser error:"));
        let page = json!({"title":"Google", "url":"https://www.google.com/search?q=films", "body":"", "results":[
            {"title":"Movies", "url":"https://www.google.com/goto?url=opaque", "snippet":"Current movies"}
        ]});
        navigation.navigate(BrowserRequest::Search("films".into()), |_| {
            parse_search_page(&page.to_string()).unwrap()
        });
        for reference in ["google-result:1", "https://www.google.com/goto?url=opaque"] {
            assert_eq!(
                navigation.navigate(BrowserRequest::Open(reference.into()), |request| {
                    let BrowserRequest::Open(url) = request else {
                        panic!("expected navigation")
                    };
                    assert_eq!(url, "https://www.google.com/goto?url=opaque");
                    "Read page".into()
                }),
                "Read page"
            );
        }
        for reference in [
            "google-result:0",
            "google-result:2",
            "https://example.com/guessed",
            "https://www.google.com/goto?url=altered",
        ] {
            navigation.navigate(BrowserRequest::Open(reference.into()), |_| {
                panic!("unknown result opened")
            });
        }
        navigation.navigate(BrowserRequest::Search("another query".into()), |_| "Search page: https://www.google.com/search?q=other\n1. Other\nURL: https://other.test/\nSnippet: Other".into());
        navigation.navigate(
            BrowserRequest::Open("https://www.google.com/goto?url=opaque".into()),
            |_| panic!("stale result opened"),
        );
    }

    #[test]
    fn a_third_search_waits_until_a_result_is_opened() {
        let search = |n: &mut SearchNavigation, query: &str| {
            n.navigate(BrowserRequest::Search(query.into()), |_| {
                "Search page: https://www.google.com/search?q=x\n1. One\nURL: https://one.test/\nSnippet: One".into()
            })
        };
        let mut navigation = SearchNavigation::default();
        assert!(search(&mut navigation, "first").starts_with("Search page: "));
        assert!(search(&mut navigation, "refined").starts_with("Search page: "));
        assert!(search(&mut navigation, "again").starts_with("Browser error: open"));
        // An invalid reference does not count as reading a result.
        navigation.navigate(BrowserRequest::Open("google-result:9".into()), |_| {
            panic!("unknown result opened")
        });
        assert!(search(&mut navigation, "again").starts_with("Browser error: open"));
        navigation.navigate(BrowserRequest::Open("google-result:1".into()), |_| {
            "Page: One\nURL: https://one.test/\n\nText".into()
        });
        assert!(search(&mut navigation, "after reading").starts_with("Search page: "));
        assert!(search(&mut navigation, "refined").starts_with("Search page: "));
        // Failed searches leave nothing to read and never block the next one.
        let mut navigation = SearchNavigation::default();
        navigation.navigate(BrowserRequest::Search("down".into()), |_| {
            "Browser error: navigation failed".into()
        });
        assert!(search(&mut navigation, "retry").starts_with("Search page: "));
        assert!(search(&mut navigation, "refined").starts_with("Search page: "));
    }

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

    #[test]
    fn search_results_show_sources_but_keep_redirects_out_of_the_prompt() {
        let page = json!({"title":"Google", "url":"https://www.google.com/search?q=rust", "body":"", "results":[
            {"title":"Rust 1.99.0", "url":"https://www.google.com/url?q=CAESopaque", "source":"blog.rust-lang.org", "snippet":"1 oct. 2026 — Rust 1.99.0 is out."},
            {"title":"Legacy card", "url":"https://example.com/legacy", "snippet":"No source field."}
        ]});
        let mut navigation = SearchNavigation::default();
        let prompt = navigation.navigate(BrowserRequest::Search("rust".into()), |_| {
            parse_search_page(&page.to_string()).unwrap()
        });
        assert!(prompt.contains(
            "1. Rust 1.99.0\nSource: blog.rust-lang.org\nSnippet: 1 oct. 2026 — Rust 1.99.0 is out."
        ));
        assert!(prompt.contains("2. Legacy card\nSnippet: No source field."));
        assert!(!prompt.contains("CAESopaque"));
    }
}
