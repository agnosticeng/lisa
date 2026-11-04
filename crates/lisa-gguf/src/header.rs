use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

use super::TensorType;

const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_ENTRIES: u64 = 1_000_000;
const MAX_VALUES: usize = 4_000_000;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Unsigned(n) => Some(*n),
            Self::Signed(n) => (*n).try_into().ok(),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(n) => Some(*n),
            Self::Unsigned(n) => Some(*n as f64),
            Self::Signed(n) => Some(*n as f64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(s) = self {
            Some(s)
        } else {
            None
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        if let Self::Array(a) = self {
            Some(a)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// GGML order: contiguous input columns first, then rows, then experts.
    pub dimensions: Vec<u64>,
    pub kind: TensorType,
    pub offset: u64,
    pub bytes: u64,
}

impl TensorInfo {
    pub fn rows(&self) -> u64 {
        // Header parsing has already checked the shape product for overflow.
        self.dimensions[1..].iter().product()
    }

    pub fn row_bytes(&self) -> Result<u64> {
        self.kind.row_bytes(self.dimensions[0])
    }
}

#[derive(Debug)]
pub struct Header {
    pub metadata: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub data_offset: u64,
    pub file_bytes: u64,
}

impl Header {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("opening GGUF {}", path.display()))?;
        let meta = file.metadata()?;
        ensure!(
            meta.is_file(),
            "GGUF is not a regular file: {}",
            path.display()
        );
        Self::read(&mut std::io::BufReader::new(file), meta.len())
            .with_context(|| format!("reading GGUF {}", path.display()))
    }

    /// `file_bytes` is the complete source's length. This also permits a
    /// bounded HTTP header download to be inspected before fetching weights.
    pub fn read(reader: &mut (impl Read + Seek), file_bytes: u64) -> Result<Self> {
        ensure!(file_bytes >= 24, "truncated GGUF header");
        reader.seek(SeekFrom::Start(0))?;
        let mut r = Reader {
            source: reader,
            pos: 0,
            limit: file_bytes.min(MAX_HEADER_BYTES),
            values: 0,
        };
        ensure!(r.bytes::<4>()? == *b"GGUF", "invalid GGUF magic");
        ensure!(r.u32()? == 3, "only GGUF version 3 is supported");
        let tensor_count = r.u64()?;
        let metadata_count = r.u64()?;
        ensure!(
            tensor_count <= MAX_ENTRIES && metadata_count <= MAX_ENTRIES,
            "GGUF header has too many entries"
        );
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = r.string()?;
            ensure!(
                !key.is_empty() && key.len() <= 65536,
                "invalid GGUF metadata key"
            );
            let kind = r.u32()?;
            let value = r.value(kind, false)?;
            ensure!(
                metadata.insert(key.clone(), value).is_none(),
                "duplicate GGUF key {key}"
            );
        }
        let alignment = match metadata.get("general.alignment") {
            None => 32,
            Some(v) => v.as_u64().context("invalid GGUF alignment type")?,
        };
        ensure!(
            alignment.is_power_of_two() && alignment <= 65536,
            "invalid GGUF alignment {alignment}"
        );
        let mut tensors = BTreeMap::new();
        for _ in 0..tensor_count {
            let name = r.string()?;
            ensure!(
                !name.is_empty() && name.len() <= 65536,
                "invalid GGUF tensor name"
            );
            let rank = r.u32()?;
            ensure!(
                (1..=4).contains(&rank),
                "invalid GGUF rank {rank} for {name}"
            );
            let mut dimensions = Vec::with_capacity(rank as usize);
            let mut elements = 1u64;
            for _ in 0..rank {
                let d = r.u64()?;
                ensure!(d > 0, "zero dimension for GGUF tensor {name}");
                elements = elements.checked_mul(d).context("GGUF shape overflow")?;
                dimensions.push(d);
            }
            let kind = TensorType(r.u32()?);
            let offset = r.u64()?;
            ensure!(offset % alignment == 0, "unaligned GGUF tensor {name}");
            let bytes = kind
                .row_bytes(dimensions[0])?
                .checked_mul(elements / dimensions[0])
                .context("GGUF tensor size overflow")?;
            let info = TensorInfo {
                name: name.clone(),
                dimensions,
                kind,
                offset,
                bytes,
            };
            ensure!(
                tensors.insert(name.clone(), info).is_none(),
                "duplicate GGUF tensor {name}"
            );
        }
        let data_offset = r
            .pos
            .checked_add(alignment - 1)
            .context("GGUF alignment overflow")?
            & !(alignment - 1);
        ensure!(data_offset <= file_bytes, "truncated GGUF header padding");
        let mut ranges = Vec::with_capacity(tensors.len());
        for t in tensors.values_mut() {
            t.offset = data_offset
                .checked_add(t.offset)
                .context("GGUF offset overflow")?;
            let end = t
                .offset
                .checked_add(t.bytes)
                .context("GGUF tensor range overflow")?;
            ensure!(
                end <= file_bytes,
                "GGUF tensor {} extends beyond file",
                t.name
            );
            ranges.push((t.offset, end, &t.name));
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            ensure!(
                pair[0].1 <= pair[1].0,
                "overlapping GGUF tensors {} and {}",
                pair[0].2,
                pair[1].2
            );
        }
        Ok(Self {
            metadata,
            tensors,
            data_offset,
            file_bytes,
        })
    }

    pub fn architecture(&self) -> Result<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(Value::as_str)
            .context("GGUF has no general.architecture")
    }

    pub fn integer(&self, key: &str) -> Result<u64> {
        self.metadata
            .get(key)
            .and_then(Value::as_u64)
            .with_context(|| format!("missing or invalid GGUF integer {key}"))
    }
}

struct Reader<'a, R> {
    source: &'a mut R,
    pos: u64,
    limit: u64,
    values: usize,
}

impl<R: Read> Reader<'_, R> {
    fn claim(&mut self, bytes: u64) -> Result<()> {
        self.pos = self
            .pos
            .checked_add(bytes)
            .context("GGUF cursor overflow")?;
        ensure!(self.pos <= self.limit, "truncated or oversized GGUF header");
        Ok(())
    }

    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.claim(N as u64)?;
        let mut data = [0; N];
        self.source.read_exact(&mut data)?;
        Ok(data)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    fn string(&mut self) -> Result<String> {
        let size = self.u64()?;
        self.claim(size)?;
        let mut data = vec![0; usize::try_from(size)?];
        self.source.read_exact(&mut data)?;
        String::from_utf8(data).context("GGUF string is not UTF-8")
    }

    fn value(&mut self, kind: u32, nested: bool) -> Result<Value> {
        self.values += 1;
        ensure!(self.values <= MAX_VALUES, "too many GGUF metadata values");
        Ok(match kind {
            0 => Value::Unsigned(u8::from_le_bytes(self.bytes()?) as u64),
            1 => Value::Signed(i8::from_le_bytes(self.bytes()?) as i64),
            2 => Value::Unsigned(u16::from_le_bytes(self.bytes()?) as u64),
            3 => Value::Signed(i16::from_le_bytes(self.bytes()?) as i64),
            4 => Value::Unsigned(self.u32()? as u64),
            5 => Value::Signed(i32::from_le_bytes(self.bytes()?) as i64),
            6 => Value::Float(f32::from_le_bytes(self.bytes()?) as f64),
            7 => {
                let value = self.bytes::<1>()?[0];
                ensure!(value <= 1, "invalid GGUF boolean");
                Value::Bool(value != 0)
            }
            8 => Value::String(self.string()?),
            9 => {
                ensure!(!nested, "nested GGUF arrays are not supported");
                let element = self.u32()?;
                ensure!(
                    element <= 12 && element != 9,
                    "invalid GGUF array element type {element}"
                );
                let count = usize::try_from(self.u64()?)?;
                ensure!(
                    count <= MAX_VALUES - self.values,
                    "oversized GGUF metadata array"
                );
                let mut values = Vec::new();
                // No untrusted count-sized reservation before reading any values.
                for _ in 0..count {
                    values.push(self.value(element, true)?);
                }
                Value::Array(values)
            }
            10 => Value::Unsigned(self.u64()?),
            11 => Value::Signed(i64::from_le_bytes(self.bytes()?)),
            12 => Value::Float(f64::from_le_bytes(self.bytes()?)),
            _ => bail!("unknown GGUF metadata type {kind}"),
        })
    }
}
