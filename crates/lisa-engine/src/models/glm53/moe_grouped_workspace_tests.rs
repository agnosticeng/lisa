use super::*;
use lisa_mlx::jit::{self, DType, Tensor};

const TOP_K: usize = 8;
const EXPERTS: usize = 12;
const HIDDEN: usize = 256;
const INTER: usize = 128;

fn expert_fixture(metadata: Dtype) -> (Array, Vec<Array>) {
    let mut arrays = Vec::new();
    let mut addresses = Vec::new();
    for expert in 0..EXPERTS {
        for (projection, input, output) in
            [(0, HIDDEN, INTER), (1, HIDDEN, INTER), (2, INTER, HIDDEN)]
        {
            let packed = (0..output * (input / 4))
                .map(|index| {
                    let mut byte = 0u8;
                    for lane in 0..4 {
                        let code = ((index * 3 + lane + expert + projection) % 4) as u8;
                        byte |= code << (lane * 2);
                    }
                    byte
                })
                .collect::<Vec<_>>();
            let groups = input / 128;
            let scales = (0..output * groups)
                .map(|index| 0.003 + ((index + expert + projection) % 7) as f32 * 0.0002)
                .collect::<Vec<_>>();
            let biases = (0..output * groups)
                .map(|index| -0.003 + ((index + projection) % 5) as f32 * 0.0001)
                .collect::<Vec<_>>();
            let weight = Array::from_slice(&packed, &[output as i32, (input / 4) as i32]);
            let scale = Array::from_slice(&scales, &[output as i32, groups as i32])
                .as_dtype(metadata)
                .unwrap();
            let bias = Array::from_slice(&biases, &[output as i32, groups as i32])
                .as_dtype(metadata)
                .unwrap();
            addresses.extend([
                weight.gpu_address().unwrap() as i64,
                scale.gpu_address().unwrap() as i64,
                bias.gpu_address().unwrap() as i64,
            ]);
            arrays.extend([weight, scale, bias]);
        }
    }
    (
        Array::from_slice(&addresses, &[(EXPERTS * 9) as i32]),
        arrays,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_chain(
    input: &Array,
    address_table: &Array,
    keepalive: &[&Tensor],
    routes: &(Array, Array, Array, Array),
    weights: &Array,
    metadata: DType,
    tokens: usize,
    workspace: Option<&GroupedMoeViews>,
) -> [Tensor; 5] {
    let device = input.t.device();
    let (route_ids, counts, work_count, work_items) = routes;
    let project = |is_up, output| {
        jit::glm53_moe_q2_mm_grouped_ds4_prefill_with_output(
            device,
            &address_table.t,
            &input.t,
            &route_ids.t,
            &counts.t,
            &work_count.t,
            &work_items.t,
            keepalive,
            None,
            metadata,
            tokens,
            TOP_K,
            EXPERTS,
            INTER,
            is_up,
            false,
            output,
        )
        .unwrap()
    };
    let gate = project(false, workspace.and_then(|views| views.gate.as_ref()));
    let up = project(true, workspace.and_then(|views| views.up.as_ref()));
    let act = jit::glm53_moe_grouped_swiglu_weight_prefill_with_output(
        device,
        &gate,
        &up,
        &weights.t,
        tokens * TOP_K,
        INTER,
        10.0,
        workspace.map(|views| &views.act),
    )
    .unwrap();
    let down = jit::glm53_moe_q2_down_grouped_ds4_prefill_with_output(
        device,
        &address_table.t,
        &act,
        &route_ids.t,
        &counts.t,
        &work_count.t,
        &work_items.t,
        keepalive,
        None,
        metadata,
        tokens,
        TOP_K,
        EXPERTS,
        HIDDEN,
        false,
        64,
        workspace.map(|views| &views.down),
    )
    .unwrap();
    let sum = lisa_mlx::models::glm53::kernels::moe_sum8(
        &Array::new(down.clone()),
        tokens,
        HIDDEN,
        &lisa_mlx::Stream::thread_local_or_default(),
    )
    .unwrap();
    // These copies are GPU commands, enqueued before the next workspace
    // overwrite. Do not read back or wait between the A/B/A submissions.
    if workspace.is_some() {
        [
            gate.copied().unwrap(),
            up.copied().unwrap(),
            act.copied().unwrap(),
            down.copied().unwrap(),
            sum.t,
        ]
    } else {
        [gate, up, act, down, sum.t]
    }
}

fn output_bits(output: &Tensor) -> Vec<u32> {
    match output.dtype() {
        DType::F32 => output
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect(),
        DType::F16 => output
            .to_vec::<half::f16>()
            .unwrap()
            .into_iter()
            .map(|value| u32::from(value.to_bits()))
            .collect(),
        dtype => panic!("unexpected grouped output dtype {dtype:?}"),
    }
}

#[test]
fn grouped_moe_workspace_reuses_all_stages_exactly_without_intermediate_waits() {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-workspace] skipping: no Metal device available");
        return;
    }
    for (metadata, jit_metadata) in [(Dtype::Float32, DType::F32), (Dtype::Bfloat16, DType::BF16)] {
        let (address_table, experts) = expert_fixture(metadata);
        let keepalive = experts.iter().map(|array| &array.t).collect::<Vec<_>>();
        let workspace = GroupedMoeWorkspace::default();
        let mut captures = Vec::new();
        let mut grown_ids = None;
        // First allocation, growth with a distinct input, then a smaller
        // return to the original input. Final A must reproduce every bit.
        for (phase, tokens) in [(0usize, 96usize), (1, 128), (0, 96)] {
            let values = (0..tokens * HIDDEN)
                .map(|index| ((index * 17 + phase * 31) % 101) as f32 * 0.002 - 0.08)
                .collect::<Vec<_>>();
            let selected = (0..tokens)
                .flat_map(|token| {
                    (0..TOP_K).map(move |rank| ((token + rank + phase * 3) % EXPERTS) as i32)
                })
                .collect::<Vec<_>>();
            let route_weights = (0..tokens * TOP_K)
                .map(|slot| 0.07 + ((slot + phase) % TOP_K) as f32 * 0.01)
                .collect::<Vec<_>>();
            let input = Array::from_slice(&values, &[tokens as i32, HIDDEN as i32]);
            let selected = Array::from_slice(&selected, &[(tokens * TOP_K) as i32]);
            let weights = Array::from_slice(&route_weights, &[(tokens * TOP_K) as i32]);
            let routes = lisa_mlx::models::glm53::kernels::moe_expert_route_map(
                &selected,
                tokens,
                TOP_K,
                EXPERTS,
                32,
                &lisa_mlx::Stream::thread_local_or_default(),
            )
            .unwrap();
            let fresh = dispatch_chain(
                &input,
                &address_table,
                &keepalive,
                &routes,
                &weights,
                jit_metadata,
                tokens,
                None,
            );
            let mut lease = workspace.buffers.lock().unwrap();
            let views = lease
                .prepare(input.t.device(), tokens * TOP_K, INTER, HIDDEN, true)
                .unwrap();
            let ids = [
                views.gate.as_ref().unwrap(),
                views.up.as_ref().unwrap(),
                &views.act,
                &views.down,
            ]
            .map(|view| view.buffer_and_layout().0.id());
            if phase == 1 {
                grown_ids = Some(ids);
            } else if let Some(expected) = grown_ids {
                assert_eq!(
                    ids, expected,
                    "smaller forwards must retain every workspace buffer"
                );
            }
            let reused = dispatch_chain(
                &input,
                &address_table,
                &keepalive,
                &routes,
                &weights,
                jit_metadata,
                tokens,
                Some(&views),
            );
            drop(lease);
            captures.push((fresh, reused));
        }
        let mut first_bits = None;
        for (iteration, (fresh, reused)) in captures.iter().enumerate() {
            let fresh_bits = fresh.each_ref().map(|output| output_bits(output));
            let reused_bits = reused.each_ref().map(|output| output_bits(output));
            assert_eq!(
                fresh_bits, reused_bits,
                "{metadata:?} iteration {iteration}: gate/up/activation/down/sum bits differ"
            );
            if iteration == 0 {
                first_bits = Some(reused_bits);
            } else if iteration == 1 {
                for (stage, (a, b)) in first_bits
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(&reused_bits)
                    .enumerate()
                {
                    assert!(
                        a.iter().zip(b).any(|(a, b)| a != b),
                        "stage {stage} must distinguish the A and B fixtures"
                    );
                }
            } else if iteration == 2 {
                assert_eq!(
                    first_bits.as_ref().unwrap(),
                    &reused_bits,
                    "return to A changed a grouped MoE output"
                );
            }
        }
    }
}
