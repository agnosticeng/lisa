//! Native runtime allocation diagnostics. These are live bytes, not RSS or peaks.

pub fn mlx_mem_line(tag: &str) {
    let mb = |r: lisa_mlx::error::Result<usize>| r.map(|b| b >> 20).unwrap_or(0);
    eprintln!(
        "[mlx-mem {tag}] active {} MiB | private {} MiB | budget {} MiB",
        mb(lisa_mlx::memory::active_memory()),
        mb(lisa_mlx::memory::private_allocated_memory()),
        lisa_mlx::memory::ram_budget() >> 20,
    );
}
