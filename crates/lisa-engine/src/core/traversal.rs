//! Generic execution of a model-owned block traversal and capture graph.
use super::graph::PreparedGraph;
use lisa_backend::Device;
use lisa_models::traversal::{Plan, Step};
pub trait Blocks {
    type Tensor: Clone;
    type State;
    fn begin(&mut self) -> anyhow::Result<Self::State>;
    fn layer(&mut self, index: usize, state: &mut Self::State) -> anyhow::Result<()>;
    fn capture(&self, state: &Self::State) -> anyhow::Result<Self::Tensor>;
    fn finish(&self, state: Self::State) -> anyhow::Result<Self::Tensor>;
}
pub struct PreparedTraversal<D: Device> {
    steps: Vec<Step>,
    join: Option<PreparedGraph<D>>,
}
impl<D: Device> PreparedTraversal<D>
where
    D::Tensor: Clone,
{
    pub fn new(device: D, plan: Plan) -> anyhow::Result<Self> {
        plan.validate().map_err(anyhow::Error::msg)?;
        let join = plan
            .capture_join
            .map(|g| PreparedGraph::new(device, g, vec![]))
            .transpose()?;
        Ok(Self {
            steps: plan.steps,
            join,
        })
    }
    pub fn run<B: Blocks<Tensor = D::Tensor>>(
        &self,
        bindings: &mut B,
    ) -> anyhow::Result<(D::Tensor, Option<D::Tensor>)> {
        let mut state = None;
        let mut result = None;
        let mut captures = Vec::new();
        for step in &self.steps {
            match *step {
                Step::Begin => state = Some(bindings.begin()?),
                Step::Layer(i) => {
                    bindings.layer(i, state.as_mut().expect("validated traversal state"))?
                }
                Step::Capture => captures
                    .push(bindings.capture(state.as_ref().expect("validated traversal state"))?),
                Step::Finish => {
                    result =
                        Some(bindings.finish(state.take().expect("validated traversal finish"))?)
                }
            }
        }
        let capture = self
            .join
            .as_ref()
            .map(|g| {
                g.run(&captures.iter().collect::<Vec<_>>(), &mut [])?
                    .into_one()
                    .map_err(anyhow::Error::msg)
            })
            .transpose()?;
        Ok((result.expect("validated traversal output"), capture))
    }
}
