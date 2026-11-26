//! Optional replay of an audited sequence of existing device operations.
//! A program preserves kernel arithmetic and allocates independent writable
//! storage on every invocation. It may decline a changed input layout before
//! encoding any work; the caller then executes its ordinary prepared graph.

/// A portable scheduling preference. The provider must prove all storage
/// dependencies before overlapping dispatches; unsupported providers retain
/// ordered execution. This never permits arithmetic or allocation alias changes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProgramScheduling {
    #[default]
    Ordered,
    DependencyParallel,
}

pub trait RecordedProgram<T>: Send + Sync {
    fn set_scheduling(&mut self, scheduling: ProgramScheduling) -> bool {
        scheduling == ProgramScheduling::Ordered
    }

    fn replay(&self, inputs: &[&T]) -> anyhow::Result<Option<Vec<T>>>;
}

pub type ProgramCapture<T> = (Vec<T>, Option<Box<dyn RecordedProgram<T>>>);
