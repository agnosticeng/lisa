//! Ordered execution of model-selected immutable graph regions.
use super::PreparedGraph;
use lisa_backend::{Device, Outputs};
use lisa_models::graph::regions::{Source, SplitGraph};

struct Part<D: Device> {
    graph: PreparedGraph<D>,
    inputs: Vec<Source>,
    release: Vec<(usize, usize)>,
}

pub struct PreparedRegions<D: Device> {
    inputs: usize,
    parts: Vec<Part<D>>,
    outputs: Vec<Source>,
}

impl<D: Device + Clone> PreparedRegions<D>
where
    D::Tensor: Clone,
{
    pub fn new(
        device: D,
        split: SplitGraph,
        weights: Vec<D::Tensor>,
        replay: &[bool],
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !split.parts.is_empty() && replay.len() == split.parts.len(),
            "invalid graph region replay flags"
        );
        let valid = |source: &Source, limit: usize| match *source {
            Source::Input(i) => i < split.input_count,
            Source::Output { part, index } => {
                part < limit && index < split.parts[part].graph.outputs.len()
            }
        };
        let mut last_use = split
            .parts
            .iter()
            .enumerate()
            .map(|(part, region)| vec![part; region.graph.outputs.len()])
            .collect::<Vec<_>>();
        for (part, region) in split.parts.iter().enumerate() {
            anyhow::ensure!(
                region.graph.inputs == region.inputs.len()
                    && region.graph.weights == region.weights.len()
                    && region.graph.states == 0
                    && region.graph.writes.is_empty()
                    && region.graph.mutable_inputs.is_empty()
                    && region.inputs.iter().all(|source| valid(source, part))
                    && region.weights.iter().all(|&i| i < weights.len()),
                "invalid graph region bindings"
            );
            for &source in &region.inputs {
                if let Source::Output {
                    part: producer,
                    index,
                } = source
                {
                    last_use[producer][index] = part;
                }
            }
        }
        anyhow::ensure!(
            !split.outputs.is_empty()
                && split
                    .outputs
                    .iter()
                    .all(|source| valid(source, split.parts.len())),
            "invalid graph region outputs"
        );
        for &source in &split.outputs {
            if let Source::Output { part, index } = source {
                last_use[part][index] = split.parts.len();
            }
        }
        let mut releases = vec![vec![]; split.parts.len()];
        for (part, uses) in last_use.iter().enumerate() {
            for (index, &last) in uses.iter().enumerate() {
                if last < releases.len() {
                    releases[last].push((part, index));
                }
            }
        }
        let parts = split
            .parts
            .into_iter()
            .enumerate()
            .map(|(part, region)| {
                let bound = region.weights.iter().map(|&i| weights[i].clone()).collect();
                let graph = PreparedGraph::new(device.clone(), region.graph, bound)?;
                Ok(Part {
                    graph: if replay[part] {
                        graph.with_dispatch_program()
                    } else {
                        graph
                    },
                    inputs: region.inputs,
                    release: std::mem::take(&mut releases[part]),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            inputs: split.input_count,
            parts,
            outputs: split.outputs,
        })
    }

    /// Set portable scheduling preferences before any region is executed.
    /// Only replay-enabled regions can accept a hint; others stay ordinary.
    pub fn with_program_scheduling(
        mut self,
        scheduling: &[lisa_backend::ProgramScheduling],
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            scheduling.len() == self.parts.len(),
            "region scheduling count differs"
        );
        for (part, &hint) in self.parts.iter_mut().zip(scheduling) {
            if let Some(program) = &mut part.graph.program {
                let program = program.get_mut().unwrap();
                anyhow::ensure!(
                    program.entries.is_empty(),
                    "region scheduling changed after capture"
                );
                program.scheduling = hint;
            }
        }
        Ok(self)
    }

    pub fn run(
        &self,
        inputs: &[&D::Tensor],
        states: &mut [D::Tensor],
    ) -> anyhow::Result<Outputs<D::Tensor>> {
        self.run_selected(inputs, states, None)
    }

    /// Select replay per invocation without changing the prepared graph or its
    /// exact-layout cache. A disabled part dispatches the same operations; a
    /// later enabled call may use the previously retained recorded layouts.
    pub fn run_with_replay(
        &self,
        inputs: &[&D::Tensor],
        states: &mut [D::Tensor],
        replay: &[bool],
    ) -> anyhow::Result<Outputs<D::Tensor>> {
        anyhow::ensure!(
            replay.len() == self.parts.len(),
            "invalid region replay selection"
        );
        self.run_selected(inputs, states, Some(replay))
    }

    fn run_selected(
        &self,
        inputs: &[&D::Tensor],
        states: &mut [D::Tensor],
        replay: Option<&[bool]>,
    ) -> anyhow::Result<Outputs<D::Tensor>> {
        anyhow::ensure!(
            inputs.len() == self.inputs && states.is_empty(),
            "invalid graph region input/state bindings"
        );
        fn resolve<'a, T>(source: Source, inputs: &[&'a T], values: &'a [Vec<Option<T>>]) -> &'a T {
            match source {
                Source::Input(i) => inputs[i],
                Source::Output { part, index } => values[part][index]
                    .as_ref()
                    .expect("live prepared region output"),
            }
        }
        let mut values = Vec::with_capacity(self.parts.len());
        for (index, part) in self.parts.iter().enumerate() {
            let run = |args: &[&D::Tensor]| {
                if replay.is_none_or(|enabled| enabled[index]) {
                    part.graph.run(args, &mut [])
                } else {
                    anyhow::ensure!(!part.graph.resources, "graph requires indirect resources");
                    part.graph.run_dispatch(args, &mut [], |handle, args| {
                        part.graph.device.dispatch(handle, args, &part.graph.stream)
                    })
                }
            };
            let output = if part.inputs.is_empty() {
                run(&[])?
            } else {
                let mut stack = [resolve(part.inputs[0], inputs, &values); 64];
                let mut heap = Vec::new();
                let args = if part.inputs.len() <= stack.len() {
                    for (i, &source) in part.inputs.iter().enumerate() {
                        stack[i] = resolve(source, inputs, &values);
                    }
                    &stack[..part.inputs.len()]
                } else {
                    heap.extend(part.inputs.iter().map(|&s| resolve(s, inputs, &values)));
                    &heap
                };
                run(args)?
            };
            values.push(output.into_iter().map(Some).collect::<Vec<_>>());
            // Earlier reads are encoded; provider leases preserve queued users.
            // Do not retain dead region temporaries for the rest of the layer.
            for &(producer, index) in &part.release {
                values[producer][index] = None;
            }
        }
        Ok(Outputs::Many(
            self.outputs
                .iter()
                .map(|&source| resolve(source, inputs, &values).clone())
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lisa_backend::{Binary, Kernel, TensorOperation};
    use lisa_models::graph::{Builder, Value, regions::split};

    #[test]
    fn regions_preserve_weight_constant_skip_and_direct_outputs() -> anyhow::Result<()> {
        let mut b = Builder::new(2, 1, 0);
        let half = b.scalar(0.5);
        let first = b.one(
            Kernel::HostTensor(TensorOperation::Binary(Binary::Add)),
            vec![Value::Input(0), Value::Weight(0)],
        );
        let middle = b.one(
            Kernel::HostTensor(TensorOperation::Binary(Binary::Multiply)),
            vec![first, half],
        );
        let last = b.one(
            Kernel::HostTensor(TensorOperation::Binary(Binary::Add)),
            vec![middle, first],
        );
        let graph = b.finish(vec![
            last,
            first,
            half,
            Value::Weight(0),
            Value::Input(1),
            last,
        ]);
        let weight = lisa_cpu::Tensor::from_f32(vec![3.0], &[1])?;
        let ordinary =
            PreparedGraph::new(lisa_cpu::CpuDevice, graph.clone(), vec![weight.clone()])?;
        let regions = PreparedRegions::new(
            lisa_cpu::CpuDevice,
            split(&graph, &[0, 1, 2, 3]).map_err(anyhow::Error::msg)?,
            vec![weight],
            &[true, false, true],
        )?;
        for x in [2.0, -3.25, 0.0] {
            let a = lisa_cpu::Tensor::from_f32(vec![x], &[1])?;
            let untouched = lisa_cpu::Tensor::from_f32(vec![7.0], &[1])?;
            let expected = ordinary.run(&[&a, &untouched], &mut [])?;
            let actual = regions.run(&[&a, &untouched], &mut [])?;
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert_eq!(actual.f32()?, expected.f32()?);
            }
        }
        Ok(())
    }
}
