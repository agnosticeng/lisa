#![cfg(test)]
// Frozen pre-migration address compositor; numerical oracle, never production.
use super::*;
use crate::core::quant::QuantizedLinearExecution;
pub trait MoEOracle1 {
    fn forward_address_gpu_oracle(
        &self,
        x: &Array,
        sel: &Array,
        wts: &Array,
        b: usize,
        t: usize,
        h: usize,
        shared_for_sum: Option<&Array>,
        prepared_resources: Option<(lisa_metal::device::IndirectTable, Vec<Array>)>,
        capture: Option<&Capture>,
        layer: usize,
    ) -> anyhow::Result<Array>;
}
impl MoEOracle1 for MoE {
    /// GPU-routed expert dispatch. The GPU reads original router expert ids
    /// and resolves each projection through the full checkpoint address table.
    fn forward_address_gpu_oracle(
        &self,
        x: &Array,
        sel: &Array,
        wts: &Array,
        b: usize,
        t: usize,
        h: usize,
        shared_for_sum: Option<&Array>,
        prepared_resources: Option<(lisa_metal::device::IndirectTable, Vec<Array>)>,
        capture: Option<&Capture>,
        layer: usize,
    ) -> anyhow::Result<Array> {
        let n = b * t;
        let k = self.top_k;
        let rows = n * k;
        if let Some(capture) = capture {
            capture.store(&format!("L{layer}.moe_input"), x);
        }
        let stream = lisa_metal::Stream::thread_local_or_default();
        let selected = sel.reshape(&[(n * k) as i32])?;
        let weights = wts.reshape(&[rows as i32])?;
        let x_rows = x.as_dtype(Dtype::Float32)?.reshape(&[n as i32, h as i32])?;
        if self.pools.gpu_address_all_experts {
            // The selector and address-based expert kernels run on the same
            // serial Metal queue. Materialize the producer without waiting:
            // queue order makes its selected IDs visible to the consumer,
            // while a host wait here serializes every routed layer.
            selected.eval_enqueue()?;
        }
        let (address_table, selected_resources) = if self.pools.gpu_address_all_experts {
            (self.pools.address_table.clone(), Vec::new())
        } else if let Some(prepared_resources) = prepared_resources {
            prepared_resources
        } else {
            let selected_ids = {
                let _trace =
                    lisa_metal::trace::span_detail("glm53.moe.selected_id_readback", layer as u64);
                selected.to_vec1_wait_through::<i32>()?
            };
            self.pools.dispatch_resources(&selected_ids)?
        };
        // A bounded cache uses the residency set only when every routed
        // buffer is already in it. Oversized prefill batches stage transient
        // experts and keep the explicit-resource path instead.
        let residency_set = self
            .pools
            .residency_set
            .get()
            .map(AsRef::as_ref)
            .filter(|_| selected_resources.is_empty());
        let indirect_expert_buffers: Vec<&Array> = selected_resources.iter().collect();
        let keepalive: &[&Array] = &indirect_expert_buffers;
        let grouped_prefill = n >= 96
            && k == 8
            && self.pools.inter.is_multiple_of(64)
            && h.is_multiple_of(128)
            && crate::core::quant::glm53_grouped_moe_prefill_override()
                .unwrap_or(n >= crate::core::quant::GLM53_GROUPED_MOE_MIN_ROWS);
        let grouped_route_map = if grouped_prefill {
            let _trace =
                lisa_metal::trace::span_detail("glm53.moe.grouped_route_map", layer as u64);
            Some(
                lisa_metal::kernels::sparse::kernels::moe_expert_route_map(
                    &selected,
                    n,
                    k,
                    self.n_experts,
                    32,
                    &stream,
                )
                .ok_or_else(|| anyhow::anyhow!("moe grouped: expert route map declined"))?,
            )
        } else {
            None
        };
        let use_vector_load =
            crate::core::quant::glm53_grouped_moe_q2_vector_load_override().unwrap_or(false);
        let use_split_pair =
            crate::core::quant::glm53_grouped_moe_pair_split_override().unwrap_or(true);
        // Keep the lease until sum8 has been encoded. The returned routed
        // tensor is independently allocated; no workspace view escapes.
        // Any capture, including preserve-dispatch capture, keeps private
        // outputs so its delayed snapshots cannot observe the next layer.
        let mut workspace_lease = if grouped_prefill && capture.is_none() {
            self.grouped_workspace
                .as_ref()
                .map(|workspace| {
                    workspace
                        .buffers
                        .lock()
                        .map_err(|_| anyhow::anyhow!("grouped MoE workspace mutex was poisoned"))
                })
                .transpose()?
        } else {
            None
        };
        let workspace = workspace_lease
            .as_mut()
            .map(|buffers| {
                buffers.prepare(
                    &crate::metal_context::device(),
                    rows,
                    self.pools.inter,
                    h,
                    use_split_pair,
                )
            })
            .transpose()?;
        let act = {
            if grouped_prefill {
                let (route_ids, counts, work_count, work_items) = grouped_route_map
                    .as_ref()
                    .expect("grouped route map was built");
                let mpp_keepalive = keepalive.iter().map(|array| &array.t).collect::<Vec<_>>();
                let metadata_dtype = match crate::metal_storage::dtype(self.pools.metadata_dtype) {
                    Dtype::Float32 => lisa_metal::jit::DType::F32,
                    Dtype::Bfloat16 => lisa_metal::jit::DType::BF16,
                    dtype => anyhow::bail!("moe grouped: unsupported metadata dtype {dtype:?}"),
                };
                let grouped = if use_split_pair {
                    let gate = {
                        let _trace = lisa_metal::trace::span_detail(
                            "glm53.moe.grouped_gate_ds4",
                            layer as u64,
                        );
                        lisa_metal::jit::moe_q2_mm_grouped_ds4_prefill_with_output(
                            x_rows.t.device(),
                            &address_table.t,
                            &x_rows.t,
                            &route_ids.t,
                            &counts.t,
                            &work_count.t,
                            &work_items.t,
                            &mpp_keepalive,
                            residency_set,
                            metadata_dtype,
                            n,
                            k,
                            self.n_experts,
                            self.pools.inter,
                            false,
                            use_vector_load,
                            workspace
                                .as_ref()
                                .and_then(|views| views.gate.as_ref().map(|v| &v.t)),
                        )?
                    };
                    let up = {
                        let _trace = lisa_metal::trace::span_detail(
                            "glm53.moe.grouped_up_ds4",
                            layer as u64,
                        );
                        lisa_metal::jit::moe_q2_mm_grouped_ds4_prefill_with_output(
                            x_rows.t.device(),
                            &address_table.t,
                            &x_rows.t,
                            &route_ids.t,
                            &counts.t,
                            &work_count.t,
                            &work_items.t,
                            &mpp_keepalive,
                            residency_set,
                            metadata_dtype,
                            n,
                            k,
                            self.n_experts,
                            self.pools.inter,
                            true,
                            use_vector_load,
                            workspace
                                .as_ref()
                                .and_then(|views| views.up.as_ref().map(|v| &v.t)),
                        )?
                    };
                    let _trace = lisa_metal::trace::span_detail(
                        "glm53.moe.grouped_swiglu_ds4",
                        layer as u64,
                    );
                    lisa_metal::jit::moe_grouped_swiglu_weight_prefill_with_output(
                        x_rows.t.device(),
                        &gate,
                        &up,
                        &weights.t,
                        n * k,
                        self.pools.inter,
                        self.limit,
                        workspace.as_ref().map(|views| &views.act.t),
                    )?
                } else {
                    let _trace = lisa_metal::trace::span_detail(
                        "glm53.moe.grouped_pair_fused_ds4",
                        layer as u64,
                    );
                    lisa_metal::jit::moe_q2_pair_grouped_ds4_prefill_with_output(
                        x_rows.t.device(),
                        &address_table.t,
                        &x_rows.t,
                        &route_ids.t,
                        &counts.t,
                        &work_count.t,
                        &work_items.t,
                        &weights.t,
                        &mpp_keepalive,
                        residency_set,
                        metadata_dtype,
                        n,
                        k,
                        self.n_experts,
                        self.pools.inter,
                        self.limit,
                        workspace.as_ref().map(|views| &views.act.t),
                    )?
                };
                Array::new(grouped)
            } else {
                let _trace = lisa_metal::trace::span_detail("glm53.moe.pair_swiglu", layer as u64);
                lisa_metal::kernels::sparse::kernels::moe_pair_swiglu_addr(
                    &address_table,
                    &x_rows,
                    &selected,
                    &weights,
                    keepalive,
                    self.pools.inter,
                    k,
                    self.limit,
                    crate::metal_storage::dtype(self.pools.metadata_dtype),
                    residency_set,
                    &stream,
                )
                .ok_or_else(|| anyhow::anyhow!("moe address: pair SwiGLU declined"))?
            }
        }; // [k, inter] F32
        if let Some(capture) = capture {
            capture.store_ephemeral(&format!("L{layer}.moe_pair_weighted"), &act);
        }

        let routed = if grouped_prefill {
            let (route_ids, counts, work_count, work_items) = grouped_route_map
                .as_ref()
                .expect("grouped route map was built");
            let metadata_dtype = match crate::metal_storage::dtype(self.pools.metadata_dtype) {
                Dtype::Float32 => lisa_metal::jit::DType::F32,
                Dtype::Bfloat16 => lisa_metal::jit::DType::BF16,
                dtype => anyhow::bail!("moe grouped down: unsupported metadata dtype {dtype:?}"),
            };
            let _trace = lisa_metal::trace::span_detail("glm53.moe.down_grouped_ds4", layer as u64);
            let down_slots = lisa_metal::jit::moe_q2_down_grouped_ds4_prefill_with_output(
                act.t.device(),
                &address_table.t,
                &act.t,
                &route_ids.t,
                &counts.t,
                &work_count.t,
                &work_items.t,
                &keepalive.iter().map(|array| &array.t).collect::<Vec<_>>(),
                residency_set,
                metadata_dtype,
                n,
                k,
                self.n_experts,
                h,
                use_vector_load,
                crate::core::quant::glm53_grouped_moe_down_tile_override().unwrap_or(64),
                workspace.as_ref().map(|views| &views.down.t),
            )?;
            let down_slots = Array::new(down_slots);
            if let Some(capture) = capture {
                capture.store_ephemeral(&format!("L{layer}.moe_down_slots"), &down_slots);
            }
            let acc = {
                let _trace =
                    lisa_metal::trace::span_detail("glm53.moe.grouped_sum8_ds4", layer as u64);
                lisa_metal::kernels::sparse::kernels::moe_sum8(&down_slots, n, h, &stream)
                    .ok_or_else(|| anyhow::anyhow!("glm53 grouped MoE: ordered sum8 declined"))?
            };

            if let Some(shared) = shared_for_sum {
                acc.add(shared)?
            } else {
                acc
            }
        } else if k == 8 && capture.is_none_or(|capture| capture.preserves_dispatch()) {
            let _trace = lisa_metal::trace::span_detail("glm53.moe.down_sum8", layer as u64);
            // Keep the reference's per-expert reduction order, add routes in order, and
            // append the shared output in the same dispatch.
            let routed = lisa_metal::kernels::sparse::kernels::moe_gather_qmv_sum8_addr_ds4(
                &address_table,
                &act,
                &selected,
                keepalive,
                shared_for_sum,
                h,
                crate::metal_storage::dtype(self.pools.metadata_dtype),
                residency_set,
                &stream,
            )
            .ok_or_else(|| anyhow::anyhow!("moe address: DS4 fused down projection declined"))?;
            if let Some(capture) = capture {
                // In diagnostic captures, compare the fused result to the
                // standalone reference matvec plus ordered sum and retain each slot.
                let down_slots = lisa_metal::kernels::sparse::kernels::moe_gather_qmv_addr_ds4(
                    &address_table,
                    &act,
                    &selected,
                    keepalive,
                    h,
                    true,
                    crate::metal_storage::dtype(self.pools.metadata_dtype),
                    residency_set,
                    &stream,
                )
                .ok_or_else(|| anyhow::anyhow!("moe address: DS4 slot capture declined"))?;
                capture.store_ephemeral(&format!("L{layer}.moe_down_slots"), &down_slots);
                let ordered =
                    lisa_metal::kernels::sparse::kernels::moe_sum8(&down_slots, n, h, &stream)
                        .ok_or_else(|| anyhow::anyhow!("glm53 MoE ordered down sum declined"))?;
                let ordered = if let Some(shared) = shared_for_sum {
                    ordered.add(shared)?
                } else {
                    ordered
                };
                let fused_values = routed.to_vec1_wait_through::<f32>()?;
                let ordered_values = ordered.to_vec1_wait_through::<f32>()?;
                anyhow::ensure!(
                    fused_values
                        .iter()
                        .map(|value| value.to_bits())
                        .eq(ordered_values.iter().map(|value| value.to_bits())),
                    "fused DS4 down projection differs from slotwise DS4 accumulation at layer {layer}"
                );
            }
            routed
        } else {
            let down_sorted = {
                let _trace = lisa_metal::trace::span_detail("glm53.moe.down_slots", layer as u64);
                lisa_metal::kernels::sparse::kernels::moe_gather_qmv_addr(
                    &address_table,
                    &act,
                    &selected,
                    keepalive,
                    h,
                    true,
                    crate::metal_storage::dtype(self.pools.metadata_dtype),
                    residency_set,
                    &stream,
                )
                .ok_or_else(|| anyhow::anyhow!("moe address: down declined"))?
            }; // [rows, h]
            if let Some(capture) = capture {
                capture.store_ephemeral(&format!("L{layer}.moe_down_slots"), &down_sorted);
            }
            let down = down_sorted.reshape(&[n as i32, k as i32, h as i32])?;
            let mut acc = narrow(&down, 1, 0, 1)?.contiguous()?;
            for slot in 1..k {
                acc = acc.add(&narrow(&down, 1, slot, 1)?.contiguous()?)?;
            }
            acc.reshape(&[n as i32, h as i32])?
        };
        // The eager sum8 dispatch above is now ordered before any later
        // layer can acquire and overwrite these four scratch buffers.
        drop(workspace_lease);
        Ok(routed.reshape(&[b as i32, t as i32, h as i32])?)
    }
}

pub trait MoEOracle0 {
    fn forward_parts_oracle(
        &self,
        x: &Array,
        capture: Option<&Capture>,
        layer: usize,
        fuse_shared_sum8: bool,
    ) -> anyhow::Result<(Array, Option<Array>)>;
}
impl MoEOracle0 for MoE {
    fn forward_parts_oracle(
        &self,
        x: &Array,
        capture: Option<&Capture>,
        layer: usize,
        fuse_shared_sum8: bool,
    ) -> anyhow::Result<(Array, Option<Array>)> {
        let (b, t, h) = (x.dim(0) as usize, x.dim(1) as usize, x.dim(2) as usize);
        let n = b * t;
        let xf = x
            .as_dtype(Dtype::Float32)?
            .reshape(&[(n) as i32, (h) as i32])?;

        // Router logits in F32. Keep the reference's token-count-specific matvec lane:
        // plain one-row, specialized two-to-eight-row, then plain per-token.
        let logits = {
            let _trace =
                lisa_metal::trace::span_detail("glm53.moe.router_projection", layer as u64);
            lisa_metal::kernels::sparse::kernels::router_matvec_f32(
                &self.gate_w,
                &xf,
                &lisa_metal::Stream::thread_local_or_default(),
            )
            .ok_or_else(|| anyhow::anyhow!("glm53 moe: ds4 F32 router matvec declined"))?
        };
        let e = self.n_experts;
        let capture = capture.filter(|capture| capture.wants(layer));
        if let Some(capture) = capture {
            capture.store(&format!("L{layer}.moe_router_logits"), &logits);
        }

        // GLM-5.3's ungrouped router and expert dispatch stay on the GPU. A
        // declined selector/address kernel is an error for this model path;
        // silently falling back to per-expert host dispatch hides a severe
        // performance and execution-path mismatch.
        if self.n_group <= 1 && self.topk_group <= 1 {
            let stream = lisa_metal::Stream::thread_local_or_default();
            let (selected, weights) = {
                let _trace =
                    lisa_metal::trace::span_detail("glm53.moe.router_select", layer as u64);
                lisa_metal::kernels::sparse::kernels::router_select_fused(
                    &logits,
                    &self.bias,
                    self.top_k,
                    self.routed_scaling,
                    &stream,
                )
                .ok_or_else(|| anyhow::anyhow!("glm53 moe: GPU router selector declined"))?
            };
            if let Some(capture) = capture {
                capture.store(&format!("L{layer}.moe_selected"), &selected);
                capture.store(&format!("L{layer}.moe_weights"), &weights);
            }

            let stage_result = if self.pools.gpu_address_all_experts {
                None
            } else {
                // Commit the selector output before the worker starts its
                // bounded readback. The shared branch can then execute in
                // later command buffers while the worker stages these experts.
                selected.eval_enqueue()?;
                self.pools.stage_selected_async(&selected)
            };

            let fuse_shared_candidate = fuse_shared_sum8
                && capture.is_none_or(|capture| capture.preserves_dispatch())
                && self.top_k == 8;
            let shared = if let Some(sh) = &self.shared {
                // Only the streaming reference needs a submission boundary
                // here. Resident execution follows the tower's reference cadence.
                if !self.pools.gpu_address_all_experts {
                    stream.runtime().commands.flush()?;
                }
                let shared = sh.forward_with_capture(x, capture, layer)?;
                if let Some(capture) = capture {
                    capture.store(&format!("L{layer}.moe_shared"), &shared);
                }
                Some(shared)
            } else {
                None
            };

            let fuse_shared = fuse_shared_candidate && shared.is_some();
            let shared_for_sum = if fuse_shared {
                shared
                    .as_ref()
                    .map(|shared| shared.as_dtype(Dtype::Float32))
                    .transpose()?
                    .map(|shared| shared.reshape(&[n as i32, h as i32]))
                    .transpose()?
            } else {
                None
            };
            let prepared_resources =
                if let Some(stage_result) = stage_result {
                    // Match the reference's streaming schedule: submit the shared-expert
                    // work before joining the selected-expert staging worker,
                    // so the GPU can execute it while the host prepares the
                    // routed experts. Without this flush, the join could wait
                    // on cache staging while the shared branch remained only
                    // encoded in the current command buffer.
                    stream.runtime().commands.flush()?;
                    let _trace = lisa_metal::trace::span("glm53.moe.async_expert_stage_join");
                    Some(stage_result.recv().map_err(|_| {
                        anyhow::anyhow!("GLM async expert staging worker stopped")
                    })??)
                } else {
                    None
                };
            // The combined shared-down/sum8 kernel misses the stock M=1 QMV
            // arithmetic on real layer-45 weights; keep shared down separate
            // until that path can satisfy the strict serial numerics contract.
            let y = self
                .forward_address_gpu(
                    x,
                    &selected,
                    &weights,
                    b,
                    t,
                    h,
                    shared_for_sum.as_ref(),
                    prepared_resources,
                    capture,
                    layer,
                )
                .map_err(|e| anyhow::anyhow!("glm53 moe: GPU address route failed: {e}"))?;
            if let Some(capture) = capture {
                capture.store(&format!("L{layer}.moe_routed"), &y);
            }
            return Ok((y, if fuse_shared { None } else { shared }));
        }

        // Grouped routing is not used by the pinned GLM-5.3 artifact. Keep its
        // explicit reference implementation for configurations that request
        // it, while the production ungrouped route above remains GPU-only.
        let logits_h: Vec<f32> = logits.to_vec1()?;
        let bias_h: Vec<f32> = self.bias.as_dtype(Dtype::Float32)?.to_vec1()?;
        let mut rows = Vec::with_capacity(n);
        for row in 0..n {
            let lg = &logits_h[row * e..(row + 1) * e];
            let sel_sc: Vec<f32> = lg
                .iter()
                .zip(bias_h.iter())
                .map(|(l, b)| 1.0 / (1.0 + (-l).exp()) + b)
                .collect();
            let per_group = e / self.n_group;
            let mut group_sc: Vec<(f32, usize)> = Vec::with_capacity(self.n_group);
            for g in 0..self.n_group {
                let lo = g * per_group;
                let mut top = [0f32; 2];
                for si in &sel_sc[lo..lo + per_group] {
                    if *si > top[0] {
                        top[1] = top[0];
                        top[0] = *si;
                    } else if *si > top[1] {
                        top[1] = *si;
                    }
                }
                group_sc.push((top[0] + top[1], g));
            }
            group_sc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut allowed = vec![false; e];
            for (_, g) in group_sc.iter().take(self.topk_group) {
                for i in (g * per_group)..((g + 1) * per_group) {
                    allowed[i] = true;
                }
            }
            let mut candidates: Vec<(f32, usize)> = (0..e)
                .filter(|&i| allowed[i])
                .map(|i| (sel_sc[i], i))
                .collect();
            candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let chosen: Vec<usize> = candidates
                .iter()
                .take(self.top_k)
                .map(|(_, i)| *i)
                .collect();
            let mut route_weights: Vec<f32> = chosen
                .iter()
                .map(|&i| 1.0 / (1.0 + (-lg[i]).exp()))
                .collect();
            if self.norm_topk {
                let sum: f32 = route_weights.iter().sum();
                if sum > 0.0 {
                    for weight in &mut route_weights {
                        *weight /= sum;
                    }
                }
            }
            for weight in &mut route_weights {
                *weight *= self.routed_scaling;
            }

            let xr = narrow(&xf, 0, row, 1)?;
            let mut acc: Option<Array> = None;
            for (slot, &expert_id) in chosen.iter().enumerate() {
                let gate_up = self.expert_gu(expert_id)?;
                let down = self.expert_down(expert_id)?;
                let fused = gate_up.forward(&xr)?;
                let (gate, up) = split_gate_up(&fused, self.pools.inter)?;
                let route_weight = Array::from_f32_as(route_weights[slot], Dtype::Float32);
                let hidden = clamped_swiglu(&up, &gate, self.limit)?.multiply(&route_weight)?;
                let term = down.forward(&hidden)?.as_dtype(Dtype::Float32)?;
                acc = Some(match acc {
                    Some(sum) => sum.add(&term)?,
                    None => term,
                });
            }
            rows.push(
                acc.ok_or_else(|| {
                    anyhow::anyhow!("glm53 moe: grouped router selected no experts")
                })?,
            );
        }
        let y = lisa_metal::ops::concatenate(&rows.iter().collect::<Vec<_>>(), 0)?
            .reshape(&[b as i32, t as i32, h as i32])?;
        let shared = self.shared.as_ref().map(|sh| sh.forward(x)).transpose()?;
        Ok((y, shared))
    }
}
