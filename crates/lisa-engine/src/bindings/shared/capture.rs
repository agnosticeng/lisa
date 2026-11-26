//! Per-module hidden-state capture for model towers. Tests attach a capture
//! to a tower to record per-layer submodule inputs and outputs for the layers
//! they select, then compare stages against an oracle.

use lisa_backend::TensorAccess;

use std::collections::HashMap;
use std::sync::Mutex;

pub struct Capture<T> {
    pub layers: Vec<usize>,
    /// Optional substring filter for targeted parity captures. Empty keeps
    /// the historical all-stages behavior for selected layers.
    pub key_filters: Vec<String>,
    pub state_only: bool,
    pub kda_prefill_only: bool,
    pub preserve_dispatch: bool,
    pub entries: Mutex<HashMap<String, Vec<f32>>>,
    pub shapes: Mutex<HashMap<String, Vec<i64>>>,
    pub pending: Mutex<HashMap<String, T>>,
    /// One-shot flush: the first forward's captures are the comparison
    /// target; later (generation re-prefill) forwards overwrite entries.
    pub flushed: std::sync::atomic::AtomicBool,
}

impl<T: TensorAccess> Capture<T> {
    pub fn wants(&self, layer: usize) -> bool {
        self.layers.contains(&layer)
    }

    pub fn preserves_dispatch(&self) -> bool {
        self.preserve_dispatch
    }

    pub fn accepts_key(&self, key: &str) -> bool {
        !self.flushed.load(std::sync::atomic::Ordering::Relaxed)
            && (self.key_filters.is_empty()
                || self.key_filters.iter().any(|filter| key.contains(filter)))
            && (!self.kda_prefill_only || key.contains(".kda_") || key.ends_with(".fused_state"))
            && (!self.state_only || key.contains("state_after_ffn_s") || key.starts_with("final_"))
    }

    /// Store a module output (flattened to F32 on host); skipped once flushed.
    pub fn store(&self, key: &str, a: &T) {
        if !self.accepts_key(key) {
            return;
        }
        let shape: Vec<i64> = a.shape().iter().map(|&d| d as i64).collect();
        if self.preserve_dispatch {
            self.pending
                .lock()
                .expect("capture lock")
                .insert(key.to_string(), a.clone());
            self.shapes
                .lock()
                .expect("capture lock")
                .insert(key.to_string(), shape);
            return;
        }
        let Ok(data) = a.read_f32() else {
            return;
        };
        self.entries
            .lock()
            .expect("capture lock")
            .insert(key.to_string(), data);
        self.shapes
            .lock()
            .expect("capture lock")
            .insert(key.to_string(), shape);
    }

    /// Preserve a transient tensor's value in a separate device buffer before
    /// later dispatches can recycle its scratch storage. The copy stays on the
    /// same command stream, so this does not add a host synchronization.
    pub fn store_ephemeral(&self, key: &str, a: &T) {
        if !self.accepts_key(key) {
            return;
        }
        if self.preserves_dispatch()
            && let Ok(snapshot) = a.independent_copy()
        {
            self.store(key, &snapshot);
        } else {
            self.store(key, a);
        }
    }

    /// Flush to npz (host-side writer; a stored-entry zip built by hand —
    /// the stdlib has no zip, the format is simple: local header + data +
    /// central directory + end record, CRC32 over the raw bytes).
    pub fn flush(&self, path: &str) -> anyhow::Result<()> {
        use std::io::Write;

        if self.preserve_dispatch {
            let pending = {
                let mut pending = self.pending.lock().expect("capture lock");
                std::mem::take(&mut *pending)
            };
            if !pending.is_empty() {
                let mut entries = self.entries.lock().expect("capture lock");
                for (key, array) in pending {
                    let Ok(values) = array.read_f32() else {
                        continue;
                    };
                    entries.insert(key, values);
                }
            }
        }

        fn crc32(data: &[u8]) -> u32 {
            // IEEE crc32, table-free bit loop (small data; fine).
            let mut crc: u32 = 0xFFFF_FFFF;
            for &b in data {
                crc ^= b as u32;
                for _ in 0..8 {
                    let mask = (crc & 1).wrapping_neg();
                    crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
                }
            }
            !crc
        }

        let entries = self.entries.lock().expect("capture lock");
        let shapes = self.shapes.lock().expect("capture lock");
        let mut names: Vec<&String> = entries.keys().collect();
        names.sort();

        let mut blob: Vec<u8> = Vec::new();
        let mut central: Vec<u8> = Vec::new();
        for name in &names {
            let data = &entries[*name];
            let shape = &shapes[*name];
            let mut npy = Vec::with_capacity(data.len() * 4 + 128);
            let header = format!(
                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}",
                shape
                    .iter()
                    .map(|d| d.to_string() + ",")
                    .collect::<String>()
            );
            // total preamble (6 magic + 2 version + 2 hlen) + header + pad + \n
            // must be a multiple of 64.
            let pad = (64 - ((10 + header.len() + 1) % 64)) % 64;
            let hlen = (header.len() + pad + 1) as u16;
            npy.extend_from_slice(&[0x93u8, b'N', b'U', b'M', b'P', b'Y', 1, 0]);
            npy.extend_from_slice(&hlen.to_le_bytes());
            npy.extend_from_slice(header.as_bytes());
            npy.extend(std::iter::repeat_n(b' ', pad));
            npy.extend_from_slice(
                b"
",
            );
            // SAFETY: `data` is a live Vec<f32>; its data.len() * 4 initialized bytes
            // are valid u8 reads (alignment 1) for the duration of this borrow.
            npy.extend_from_slice(unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
            });

            let fname = format!("{name}.npy");
            let crc = crc32(&npy);
            let off = blob.len() as u32;
            // local file header
            blob.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            blob.extend_from_slice(&20u16.to_le_bytes()); // version
            blob.extend_from_slice(&0u16.to_le_bytes()); // flags
            blob.extend_from_slice(&0u16.to_le_bytes()); // stored
            blob.extend_from_slice(&0u16.to_le_bytes()); // time
            blob.extend_from_slice(&0u16.to_le_bytes()); // date
            blob.extend_from_slice(&crc.to_le_bytes());
            blob.extend_from_slice(&(npy.len() as u32).to_le_bytes());
            blob.extend_from_slice(&(npy.len() as u32).to_le_bytes());
            blob.extend_from_slice(&(fname.len() as u16).to_le_bytes());
            blob.extend_from_slice(&0u16.to_le_bytes()); // extra len
            blob.extend_from_slice(fname.as_bytes());
            blob.extend_from_slice(&npy);
            // central directory entry
            central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            central.extend_from_slice(&[20, 0]); // version made by
            central.extend_from_slice(&20u16.to_le_bytes()); // version needed
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&crc.to_le_bytes());
            central.extend_from_slice(&(npy.len() as u32).to_le_bytes());
            central.extend_from_slice(&(npy.len() as u32).to_le_bytes());
            central.extend_from_slice(&(fname.len() as u16).to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes()); // extra
            central.extend_from_slice(&0u16.to_le_bytes()); // comment
            central.extend_from_slice(&0u16.to_le_bytes()); // disk
            central.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
            central.extend_from_slice(&0u32.to_le_bytes()); // external attrs
            central.extend_from_slice(&off.to_le_bytes());
            central.extend_from_slice(fname.as_bytes());
        }
        let cd_off = blob.len() as u32;
        let cd_len = central.len() as u32;
        blob.extend_from_slice(&central);
        // end of central directory
        blob.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        blob.extend_from_slice(&0u16.to_le_bytes()); // disk
        blob.extend_from_slice(&0u16.to_le_bytes()); // cd disk
        blob.extend_from_slice(&(names.len() as u16).to_le_bytes());
        blob.extend_from_slice(&(names.len() as u16).to_le_bytes());
        blob.extend_from_slice(&cd_len.to_le_bytes());
        blob.extend_from_slice(&cd_off.to_le_bytes());
        blob.extend_from_slice(&0u16.to_le_bytes()); // comment len

        std::fs::File::create(path)?.write_all(&blob)?;
        Ok(())
    }
}

impl<T> Default for Capture<T> {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            key_filters: Vec::new(),
            state_only: false,
            kda_prefill_only: false,
            preserve_dispatch: false,
            entries: Mutex::new(HashMap::new()),
            shapes: Mutex::new(HashMap::new()),
            pending: Mutex::new(HashMap::new()),
            flushed: std::sync::atomic::AtomicBool::new(false),
        }
    }
}
