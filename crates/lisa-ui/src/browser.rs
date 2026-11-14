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
