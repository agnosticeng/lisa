//! Process-global MLX allocator diagnostics.

pub fn mlx_mem_line(tag: &str) {
    let mb = |r: lisa_mlx::error::Result<usize>| r.map(|b| b >> 20).unwrap_or(0);
    eprintln!(
        "[mlx-mem {tag}] active {} MB | cache {} MB | peak {} MB | limit {} MB",
        mb(lisa_mlx::memory::active_memory()),
        mb(lisa_mlx::memory::cache_memory()),
        mb(lisa_mlx::memory::peak_memory()),
        mb(lisa_mlx::memory::memory_limit()),
    );
}
