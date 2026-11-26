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
#[derive(Default)]
pub(crate) struct SearchNavigation {
    results: Vec<String>,
}

impl SearchNavigation {
    pub(crate) fn navigate(
        &mut self,
        request: BrowserRequest,
        navigate: impl FnOnce(BrowserRequest) -> String,
    ) -> String {
        match request {
            BrowserRequest::Search(query) => {
                let content = navigate(BrowserRequest::Search(query));
                if content.starts_with("Search page: ") {
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
                    Some(url) => navigate(BrowserRequest::Open(url.clone())),
                    None => "Browser error: choose a result number from the latest Google search. Only Google result links can be opened; do not invent a URL.".into(),
                }
            }
        }
    }
}

/// Web answers contain source text, not unchecked model-written claims. The
/// browser's formatted page header supplies provenance; body URLs do not.
#[derive(Default)]
pub(crate) struct ReadPages {
    pages: Vec<(String, String, String)>,
}

impl ReadPages {
    pub(crate) fn remember(&mut self, content: &str) {
        let Some((header, body)) = content.split_once("\n\n") else {
            return;
        };
        let mut lines = header.lines();
        let Some(title) = lines.next().and_then(|line| line.strip_prefix("Page: ")) else {
            return;
        };
        let Some(url) = lines.next().and_then(|line| line.strip_prefix("URL: ")) else {
            return;
        };
        if self.pages.iter().any(|(_, seen, _)| seen == url) {
            return;
        }
        let body = body
            .split("\n\nLinks on this page:")
            .next()
            .unwrap_or(body)
            .trim();
        if body.is_empty() || !(url.starts_with("https://") || url.starts_with("http://")) {
            return;
        }
        self.pages.push((title.into(), url.into(), body.into()));
    }

    pub(crate) fn answer(&self) -> Option<String> {
        if self.pages.is_empty() {
            return None;
        }
        Some(
            self.pages
                .iter()
                .take(3)
                .map(|(title, url, body)| {
                    let title = title.replace('[', "\\[").replace(']', "\\]");
                    // Release listings expose a title followed by a dated metadata row.
                    // Render those exact fields, excluding ratings and site controls.
                    let lines = body
                        .lines()
                        .map(str::trim)
                        .filter(|line| !line.is_empty())
                        .collect::<Vec<_>>();
                    let mut entries = lines
                        .windows(2)
                        .filter_map(|pair| {
                            if let Some(date) = pair[1].strip_prefix("Sortie le ")
                                && pair[0].chars().count() <= 180
                                && date.len() <= 80
                            {
                                return Some(format!(
                                    "- **{}**  \n  {}",
                                    escape_markdown(pair[0]),
                                    escape_markdown(date)
                                ));
                            }
                            let fields = pair[1].split('|').map(str::trim).collect::<Vec<_>>();
                            if fields.len() < 2 || pair[0].chars().count() > 180 {
                                return None;
                            }
                            let date = fields[0];
                            let has_year =
                                date.split(|ch: char| !ch.is_ascii_digit()).any(|part| {
                                    part.len() == 4
                                        && part
                                            .parse::<u32>()
                                            .is_ok_and(|year| (1900..=2199).contains(&year))
                                });
                            if !has_year || date.len() > 64 {
                                return None;
                            }
                            let detail = if fields.len() >= 3 && !fields.last().unwrap().is_empty()
                            {
                                format!("{} · {}", date, fields.last().unwrap())
                            } else {
                                date.into()
                            };
                            Some(format!(
                                "- **{}**  \n  {}",
                                escape_markdown(pair[0]),
                                escape_markdown(&detail)
                            ))
                        })
                        .take(8)
                        .collect::<Vec<_>>();
                    if entries.is_empty() {
                        // Some listings place title and release date on the same row.
                        entries = lines
                            .iter()
                            .filter_map(|line| {
                                let (name, date) = line.split_once("Sortie le ")?;
                                let name = name.trim();
                                if name.is_empty() || name.chars().count() > 180 || date.len() > 80
                                {
                                    return None;
                                }
                                Some(format!(
                                    "- **{}**  \n  {}",
                                    escape_markdown(name),
                                    escape_markdown(date.trim())
                                ))
                            })
                            .take(8)
                            .collect();
                    }
                    if !entries.is_empty() {
                        return format!("{}\n\n[{title}](<{url}>)", entries.join("\n\n"));
                    }
                    let excerpt = body.chars().take(500).collect::<String>();
                    let truncated = excerpt.len() < body.len();
                    let excerpt = if truncated {
                        // Do not finish a truncated excerpt halfway through a word.
                        excerpt
                            .rsplit_once(char::is_whitespace)
                            .map(|(text, _)| text)
                            .unwrap_or(&excerpt)
                            .to_string()
                    } else {
                        excerpt
                    };
                    // Escape markup from pages so quoted text cannot create invented
                    // links or alter the source card's Markdown structure.
                    let mut quote = excerpt
                        .lines()
                        .map(|line| {
                            let escaped = escape_markdown(line);
                            format!("> {escaped}")
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if truncated {
                        quote.push_str("\n> […]");
                    }
                    format!("[{title}](<{url}>)\n\n{quote}")
                })
                .collect::<Vec<_>>()
                .join("\n\n"),
        )
    }
}

fn escape_markdown(text: &str) -> String {
    text.chars()
        .flat_map(|ch| {
            if "\\`*_{}[]<>()#+-.!|".contains(ch) {
                vec!['\\', ch]
            } else {
                vec![ch]
            }
        })
        .collect()
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
  // Mobile results use opaque /goto redirects; WebKit resolves them on open.
  const redirects = new Set(['/url', '/goto']);
  const results = Array.from(document.querySelectorAll('a'))
    .map(a => {
      const heading = a.querySelector('h3, [role="heading"][aria-level="3"]');
      if (!heading) return null;
      let url = a.href;
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
      } catch (_) { return null; }
      let card = a;
      for (let i = 0; i < 5 && card.parentElement; i++) {
        card = card.parentElement;
        const text = clean(card.textContent);
        if (text.length > clean(heading.innerText || heading.textContent).length + 20 && text.length < 1400) break;
      }
      const title = clean(heading.innerText || heading.textContent);
      if (!title) return null;
      const text = clean(card.textContent);
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
    fn web_answers_only_render_read_page_excerpts() {
        let mut pages = ReadPages::default();
        pages.remember("Browser error: navigation failed");
        pages.remember("Search page: https://google.com\nURL: https://example.com");
        assert!(pages.answer().is_none());
        pages.remember("Page: New films\nURL: https://example.com/films\n\nA real film opens on Thursday.\n[Invented link](https://invented.test)\n\nLinks on this page:\n- Navigation — https://other.test");
        let answer = pages.answer().unwrap();
        assert!(answer.starts_with("[New films](<https://example.com/films>)"));
        assert!(answer.contains("> A real film opens on Thursday\\."));
        assert!(answer.contains("\\[Invented link\\]"));
        assert!(!answer.contains("https://other.test"));
        assert!(!answer.contains("Browser error"));
    }

    #[test]
    fn release_listings_show_source_titles_dates_and_genres_without_site_controls() {
        let mut pages = ReadPages::default();
        pages.remember("Page: This week's releases\nURL: https://example.com/releases\n\nHome\nNew films\nFirst Film\n18 novembre 2026 | 2h 38min | Action, Adventure\nRatings\n3,8\nBOOK TICKETS\nSecond Film\n18 novembre 2026 | 1h 31min | Comedy\nLong description.");
        let answer = pages.answer().unwrap();
        assert!(answer.contains("- **First Film**  \n  18 novembre 2026 · Action, Adventure"));
        assert!(answer.contains("- **Second Film**  \n  18 novembre 2026 · Comedy"));
        assert!(answer.ends_with("[This week's releases](<https://example.com/releases>)"));
        for noise in [
            "Ratings",
            "BOOK TICKETS",
            "Long description",
            "Home",
            "2h 38min",
        ] {
            assert!(!answer.contains(noise));
        }
        let mut inline = ReadPages::default();
        inline.remember("Page: Releases\nURL: https://example.com/releases\n\nNavigation\nFirst Film Sortie le Mercredi 18 novembre 2026\nSecond Film Sortie le Mercredi 11 novembre 2026");
        let answer = inline.answer().unwrap();
        assert!(answer.contains("- **First Film**  \n  Mercredi 18 novembre 2026"));
        assert!(answer.contains("- **Second Film**  \n  Mercredi 11 novembre 2026"));
        assert!(!answer.contains("Navigation"));
        let mut multiline = ReadPages::default();
        multiline.remember("Page: Releases\nURL: https://example.com/releases\n\nNavigation\nFirst Film\nSortie le Mercredi 18 novembre 2026\n  \nSecond Film\nSortie le Mercredi 11 novembre 2026");
        assert_eq!(multiline.answer(), inline.answer());
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
}
