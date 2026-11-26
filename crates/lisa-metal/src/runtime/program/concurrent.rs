//! Static buffer hazards for an explicitly selected recorded-program schedule.
//! Arithmetic and dispatch order are unchanged. A barrier separates conflicting
//! physical ranges; independent dispatches in the same epoch may overlap.
use super::{Command, ScratchArena, Section, Storage};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
struct Access {
    domain: usize,
    begin: usize,
    end: usize,
    write: bool,
}
impl Access {
    fn conflicts(self, other: Self) -> bool {
        self.domain == other.domain
            && self.begin < other.end
            && other.begin < self.end
            && (self.write || other.write)
    }
}

#[derive(Default)]
struct Epoch {
    accesses: Vec<Access>,
    dispatches: usize,
    independent: usize,
}
impl Epoch {
    fn push(&mut self, next: &[Access]) -> bool {
        let barrier = next
            .iter()
            .any(|&a| self.accesses.iter().any(|&b| a.conflicts(b)));
        if barrier {
            self.accesses.clear();
            self.dispatches = 0;
        } else if self.dispatches > 0 {
            self.independent += 1;
        }
        self.accesses.extend_from_slice(next);
        self.dispatches += 1;
        barrier
    }
    fn boundary(&mut self) {
        self.accesses.clear();
        self.dispatches = 0;
    }
}

pub(super) struct ParallelPlan {
    pub(super) barriers: Vec<bool>,
    pub(super) independent_dispatches: usize,
}
impl ParallelPlan {
    pub(super) fn plan(
        storage: &[Storage],
        sections: &[Section],
        scratch: Option<&ScratchArena>,
    ) -> Option<Self> {
        let mut ranges = Vec::with_capacity(storage.len());
        for (slot, storage) in storage.iter().enumerate() {
            let bytes = match storage {
                Storage::Fresh { bytes, .. } => (*bytes).max(1).checked_add(1023)? / 1024 * 1024,
                Storage::Input { minimum_length, .. } => *minimum_length,
                Storage::Pinned(buffer) => buffer.length(),
            };
            // Packed scratch aliases are physical, even when their original
            // logical slots differ. Use whole rounded allocations, never a
            // binding offset as an unproved approximation of a shader's reads.
            let packed = scratch.and_then(|s| s.offsets[slot]);
            let (domain, begin) = packed.map_or((slot.checked_add(1)?, 0), |base| (0, base));
            ranges.push(Access {
                domain,
                begin,
                end: begin.checked_add(bytes)?,
                write: false,
            });
        }
        let mut epoch = Epoch::default();
        let mut barriers = Vec::new();
        for section in sections {
            match section {
                Section::Allocate(_) => {}
                Section::Boundary => epoch.boundary(),
                Section::Encoder(commands) => {
                    // Admitted sections are self-contained: each selects its
                    // pipeline and binds every used operand. Earlier sections'
                    // unbound argument indices are therefore unused. Within a
                    // section, inheritance across dispatches/pipeline changes
                    // remains live until Bytes/null explicitly replaces it.
                    let mut bound = HashMap::<usize, Access>::new();
                    let mut pipeline = false;
                    for command in commands {
                        match command {
                            Command::Pipeline(_) => pipeline = true,
                            Command::Buffer {
                                index, slot, write, ..
                            } => {
                                if let Some(slot) = slot {
                                    bound.insert(
                                        *index,
                                        Access {
                                            write: *write,
                                            ..*ranges.get(*slot)?
                                        },
                                    );
                                } else {
                                    bound.remove(index);
                                }
                            }
                            Command::Bytes { index, .. } => {
                                bound.remove(index);
                            }
                            Command::Threadgroup { .. } => {}
                            Command::Dispatch { .. } => {
                                if !pipeline {
                                    return None;
                                }
                                let current = bound.values().copied().collect::<Vec<_>>();
                                barriers.push(epoch.push(&current));
                            }
                        }
                    }
                }
            }
        }
        // A union with the packed plan is conservative when arena allocation
        // falls back to independent buffers: only unnecessary barriers remain.
        // Ordinary ordered reuse is excluded by the caller.
        (epoch.independent > 0).then_some(Self {
            barriers,
            independent_dispatches: epoch.independent,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn access(domain: usize, begin: usize, end: usize, write: bool) -> Access {
        Access {
            domain,
            begin,
            end,
            write,
        }
    }
    #[test]
    fn parallel_hazards_distinguish_raw_war_waw_and_disjoint_packed_ranges() {
        for (first_write, second_write) in
            [(true, false), (false, true), (true, true), (false, false)]
        {
            let mut epoch = Epoch::default();
            assert!(!epoch.push(&[access(0, 0, 1024, first_write)]));
            assert_eq!(
                epoch.push(&[access(0, 512, 1536, second_write)]),
                first_write || second_write
            );
        }
        let mut epoch = Epoch::default();
        assert!(!epoch.push(&[access(0, 0, 1024, true)]));
        assert!(!epoch.push(&[access(0, 1024, 2048, true)]));
        assert_eq!(epoch.independent, 1);
        assert!(epoch.push(&[access(0, 0, 1024, false)]));
        // A different logical scratch slot reusing the first physical range
        // cannot overwrite it until the immediately preceding read finishes.
        assert!(epoch.push(&[access(0, 0, 1024, true)]));
        assert!(!epoch.push(&[access(1, 0, 1024, true)]));
        epoch.boundary();
        assert!(!epoch.push(&[access(0, 0, 1024, false)]));
    }
    #[test]
    fn parallel_hazard_epochs_never_contain_a_conflicting_pair_exhaustively() {
        let choices = (0..2)
            .flat_map(|domain| {
                (0..3).flat_map(move |begin| {
                    (begin + 1..=3).flat_map(move |end| {
                        [false, true].map(move |write| access(domain, begin, end, write))
                    })
                })
            })
            .collect::<Vec<_>>();
        for &a in &choices {
            for &b in &choices {
                for &c in &choices {
                    let mut planner = Epoch::default();
                    let mut observed = Vec::<Access>::new();
                    for next in [a, b, c] {
                        let barrier = planner.push(&[next]);
                        if barrier {
                            observed.clear();
                        }
                        for prior in &observed {
                            let overlap = next.domain == prior.domain
                                && next.begin < prior.end
                                && prior.begin < next.end;
                            assert!(!overlap || !(next.write || prior.write));
                        }
                        observed.push(next);
                    }
                }
            }
        }
    }
}
