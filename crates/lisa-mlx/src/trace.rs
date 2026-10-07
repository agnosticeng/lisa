//! Lightweight hierarchical tracing for performance work.
//!
//! Off by default and near-zero cost (one relaxed atomic load per `span()`).
//! Enabled with `LISA_TRACE=1`; `LISA_TRACE_JSON=<path>` additionally writes a
//! Chrome Trace Event file (`chrome://tracing`, or `perfetto.dev`) with every
//! span as an `X` event, so per-layer / per-kernel timelines can be inspected
//! visually.
//!
//! Usage:
//! ```ignore
//! let _s = crate::trace::span("tower.forward");
//! ```
//! Spans nest per thread; `report()` aggregates by name (calls, total, self
//! — total minus direct children) and folds in the GPU counters from
//! [`crate::runtime`]. `finish()` is the process-exit hook: prints the report
//! when `LISA_TRACE` is set and writes the JSON when `LISA_TRACE_JSON` is set.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);
static INIT: AtomicBool = AtomicBool::new(false);
static T0: Mutex<Option<Instant>> = Mutex::new(None);

/// One closed span. `parent` indexes [`RECORDS`]; `detail` carries an optional
/// numeric discriminator (layer index, token count, …) for the Chrome view.
struct Record {
    name: &'static str,
    detail: u64,
    start_us: u64,
    dur_us: u64,
    depth: u32,
    parent: Option<u32>,
    tid: u64,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static RECORDS: Mutex<Vec<Record>> = Mutex::new(Vec::new());

std::thread_local! {
    static STACK: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    static TID: u64 = NEXT_ID.fetch_add(1, Ordering::Relaxed);
}

fn now_us() -> u64 {
    let mut t0 = T0.lock().unwrap();
    let t0 = t0.get_or_insert_with(Instant::now);
    t0.elapsed().as_micros() as u64
}

fn init_from_env() {
    if INIT.swap(true, Ordering::Relaxed) {
        return;
    }
    if std::env::var_os("LISA_TRACE").is_some() {
        ENABLED.store(true, Ordering::Relaxed);
    }
    if std::env::var_os("LISA_TRACE_JSON").is_some() {
        ENABLED.store(true, Ordering::Relaxed);
    }
}

/// Whether tracing is on (`LISA_TRACE=1` or `LISA_TRACE_JSON` set).
pub fn enabled() -> bool {
    init_from_env();
    ENABLED.load(Ordering::Relaxed)
}

/// A tracing span. Drop closes it. Cheap no-op when tracing is off.
pub struct Span {
    inner: Option<(&'static str, u64, u64, usize, u64)>, // (name, detail, start_us, record slot, tid)
}

impl Span {
    /// A disabled span (tracing off).
    pub fn none() -> Self {
        Span { inner: None }
    }
}

/// Open a span named `name` (static str; no allocation on the hot path).
pub fn span(name: &'static str) -> Span {
    span_detail(name, 0)
}

/// Open a span with a numeric detail (e.g. layer index) for the Chrome view.
pub fn span_detail(name: &'static str, detail: u64) -> Span {
    if !enabled() {
        return Span { inner: None };
    }
    let tid = TID.with(|t| *t);
    let start_us = now_us();
    let slot;
    {
        let mut recs = RECORDS.lock().unwrap();
        slot = recs.len();
        let depth = STACK.with(|s| s.borrow().len() as u32);
        let parent = STACK.with(|s| s.borrow().last().copied()).map(|p| p as u32);
        recs.push(Record {
            name,
            detail,
            start_us,
            dur_us: 0,
            depth,
            parent,
            tid,
        });
    }
    STACK.with(|s| s.borrow_mut().push(slot));
    Span {
        inner: Some((name, detail, start_us, slot, tid)),
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some((_name, _detail, start_us, slot, _tid)) = self.inner.take() {
            let dur_us = now_us().saturating_sub(start_us);
            STACK.with(|s| s.borrow_mut().pop());
            if let Some(r) = RECORDS.lock().unwrap().get_mut(slot) {
                r.dur_us = dur_us;
            }
        }
    }
}

/// Record a point event (zero duration) — e.g. phase boundaries.
pub fn event(name: &'static str) {
    if enabled() {
        let tid = TID.with(|t| *t);
        let depth = STACK.with(|s| s.borrow().len() as u32);
        RECORDS.lock().unwrap().push(Record {
            name,
            detail: 0,
            start_us: now_us(),
            dur_us: 0,
            depth,
            parent: None,
            tid,
        });
    }
}

struct Agg {
    calls: u64,
    total_us: u64,
    child_us: u64,
    min_us: u64,
    max_us: u64,
    depth: u32,
}

/// Print the aggregated report to stderr: per-name calls / total / self / avg,
/// sorted by self time, plus the GPU counters from the runtime.
pub fn report() {
    if !enabled() {
        return;
    }
    let recs = RECORDS.lock().unwrap();
    // Self time: total minus the sum of direct children.
    let mut child_sum: Vec<u64> = vec![0; recs.len()];
    for (_i, r) in recs.iter().enumerate() {
        if let Some(p) = r.parent {
            child_sum[p as usize] += r.dur_us;
        }
    }
    let mut aggs: Vec<(&'static str, Agg)> = Vec::new();
    for (i, r) in recs.iter().enumerate() {
        if r.dur_us == 0 && r.parent.is_none() && r.depth > 0 {
            continue; // point events: skip from the table
        }
        let e = aggs.iter_mut().find(|(n, _)| *n == r.name);
        let self_us = r.dur_us.saturating_sub(child_sum[i]);
        match e {
            Some((_, a)) => {
                a.calls += 1;
                a.total_us += r.dur_us;
                a.child_us += self_us;
                a.min_us = a.min_us.min(r.dur_us);
                a.max_us = a.max_us.max(r.dur_us);
            }
            None => aggs.push((
                r.name,
                Agg {
                    calls: 1,
                    total_us: r.dur_us,
                    child_us: self_us,
                    min_us: r.dur_us,
                    max_us: r.dur_us,
                    depth: r.depth,
                },
            )),
        }
    }
    aggs.sort_by(|a, b| b.1.child_us.cmp(&a.1.child_us));
    eprintln!(
        "[trace] {:<22} {:>8} {:>11} {:>11} {:>10} {:>10}",
        "name", "calls", "total ms", "self ms", "avg us", "max us"
    );
    for (name, a) in &aggs {
        eprintln!(
            "[trace] {:<22} {:>8} {:>11.1} {:>11.1} {:>10.1} {:>10.1}",
            format!("{}{}", "  ".repeat(a.depth as usize), name),
            a.calls,
            a.total_us as f64 / 1e3,
            a.child_us as f64 / 1e3,
            a.total_us as f64 / a.calls as f64,
            a.max_us as f64,
        );
    }
    eprintln!(
        "[trace] gpu: dispatches={} encoders={} wait_ms={:.0} allocs={} poolhits={}",
        crate::runtime_dispatch_count(),
        crate::runtime_encoder_count(),
        crate::runtime_wait_ns() as f64 / 1e6,
        crate::runtime_alloc_count(),
        crate::runtime_poolhit_count(),
    );
    eprintln!("[trace] jit compiles: {}", crate::runtime::jit_compiles());
    // Every kernel dispatched, with its count. This is the census source: a
    // kernel absent here never ran, which the GPU-ms table alone cannot say
    // (a kernel whose time rounds to zero is indistinguishable from one that
    // was never encoded).
    let mut ld = crate::runtime_label_dispatches();
    if !ld.is_empty() {
        ld.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!("[trace] kernels dispatched: {:?}", &ld[..]);
    }
    let mut lc = crate::runtime_label_counts();
    if !lc.is_empty() {
        lc.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!("[trace] kernels: {:?}", &lc[..]);
    }
    let mut lg = crate::runtime_label_gpu_ms();
    if !lg.is_empty() {
        lg.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!(
            "[trace] kernels by gpu ms: {:?}",
            &lg[..]
                .iter()
                .map(|(l, ns)| (l.clone(), *ns as f64 / 1e6))
                .collect::<Vec<(String, f64)>>()
        );
    }
    probe_report();
}

/// GPU-probe analysis (`LISA_GPU_PROBE=1`): per-kernel real execution
/// distributions plus the inter-kernel gap total on the GPU timeline.
fn probe_report() {
    let recs = crate::runtime::GPU_PROBE_RECORDS
        .lock()
        .unwrap()
        .clone();
    if recs.is_empty() {
        return;
    }
    // Execution distribution per label (µs).
    let mut by_label: Vec<(String, Vec<f64>)> = Vec::new();
    for (label, s, e) in &recs {
        let dur_us = (e - s).max(0.0) * 1e6;
        match by_label.iter_mut().find(|(l, _)| l == label) {
            Some((_, v)) => v.push(dur_us),
            None => by_label.push((label.clone(), vec![dur_us])),
        }
    }
    let total_exec_us: f64 = by_label.iter().map(|(_, v)| v.iter().sum::<f64>()).sum();
    let n_dispatches: usize = by_label.iter().map(|(_, v)| v.len()).sum();
    eprintln!(
        "[probe] dispatches={} total exec ms={:.2}",
        n_dispatches,
        total_exec_us / 1e3
    );
    by_label.sort_by(|a, b| {
        let sa: f64 = b.1.iter().sum();
        let s: f64 = a.1.iter().sum();
        sa.total_cmp(&s)
    });
    eprintln!(
        "[probe] {:<24} {:>7} {:>9} {:>9} {:>9} {:>9}",
        "kernel", "n", "p50 us", "p90 us", "tot ms", "us/dis"
    );
    for (label, v) in &by_label[..40.min(by_label.len())] {
        let mut sorted = v.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let p = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
        let tot: f64 = v.iter().sum();
        eprintln!(
            "[probe] {:<24} {:>7} {:>9.1} {:>9.1} {:>9.2} {:>9.1}",
            label,
            v.len(),
            p(0.5),
            p(0.9),
            tot / 1e3,
            tot / v.len() as f64
        );
    }
    // Gap accounting on the GPU timeline: sort by start; gap = next.start -
    // prev.end (>= 0), overlap ignored (single serial queue).
    let mut spans: Vec<(f64, f64)> = recs.iter().map(|(_, s, e)| (*s, *e)).collect();
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut gaps: Vec<f64> = Vec::new();
    for w in spans.windows(2) {
        let g = (w[1].0 - w[0].1).max(0.0) * 1e6;
        if g > 0.1 {
            gaps.push(g);
        }
    }
    let total_gap: f64 = gaps.iter().sum();
    let span_us = (spans.last().unwrap().1 - spans[0].0).max(0.0) * 1e6;
    let mut sorted_gaps = gaps.clone();
    sorted_gaps.sort_by(|a, b| a.total_cmp(b));
    let gp = |q: f64| {
        sorted_gaps
            .get((((sorted_gaps.len() - 1) as f64) * q) as usize)
            .copied()
            .unwrap_or(0.0)
    };
    eprintln!(
        "[probe] timeline span ms={:.2} exec ms={:.2} gap ms={:.2} ({:.0}%) gaps>0.1us={} p50={:.1}us p90={:.1}us",
        span_us / 1e3,
        total_exec_us / 1e3,
        total_gap / 1e3,
        100.0 * total_gap / span_us.max(1.0),
        gaps.len(),
        gp(0.5),
        gp(0.9)
    );
}

/// Exit hook: print the report when `LISA_TRACE` is set, write the Chrome
/// JSON when `LISA_TRACE_JSON=<path>` is set. Call once at process end.
pub fn finish() {
    init_from_env();
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    report();
    if let Some(path) = std::env::var_os("LISA_TRACE_JSON") {
        match write_chrome(&path.to_string_lossy()) {
            Ok(()) => eprintln!("[trace] chrome trace written to {path:?}"),
            Err(e) => eprintln!("[trace] chrome trace write failed: {e}"),
        }
    }
}

/// Write every record as a Chrome Trace Event `X` (complete) event.
pub fn write_chrome(path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let recs = RECORDS.lock().unwrap();
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"{\"traceEvents\":[")?;
    let mut first = true;
    for r in recs.iter() {
        if !first {
            f.write_all(b",")?;
        }
        first = false;
        let name = if r.detail != 0 {
            format!("{}#{}", r.name, r.detail)
        } else {
            r.name.to_string()
        };
        write!(
            f,
            "{{\"name\":{:?},\"cat\":\"lisa\",\"ph\":\"X\",\"ts\":{},\"dur\":{},\"pid\":1,\"tid\":{}}}",
            name, r.start_us, r.dur_us, r.tid
        )?;
    }
    f.write_all(b"]}\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_spans_are_noops() {
        // The env may enable tracing in the test process; both paths must work.
        let s = span("test.disabled");
        drop(s);
        let s = span("test.enabled");
        drop(s);
    }

    #[test]
    fn chrome_json_writes_events() {
        // Spans are no-ops unless tracing is enabled; force it for this test.
        INIT.store(true, Ordering::Relaxed);
        ENABLED.store(true, Ordering::Relaxed);
        let path = std::env::temp_dir().join("lisa_trace_test.json");
        let _ = std::fs::remove_file(&path);
        {
            let _s = span("test.chrome");
            let _inner = span("test.chrome.child");
        }
        write_chrome(path.to_str().unwrap()).unwrap();
        let txt = std::fs::read_to_string(&path).unwrap();
        assert!(txt.contains("\"ph\":\"X\""));
        assert!(txt.contains("test.chrome.child"));
        let _ = std::fs::remove_file(&path);
    }
}
