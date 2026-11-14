//! Individual indirect affine operations and provider-owned storage leases.
use super::*;
use crate::kernels::sparse::kernels as raw;

pub struct IndirectResources {
    runtime: Arc<crate::runtime::MetalRuntime>,
    table: Array,
    owners: Vec<Array>,
    residency: Option<Arc<crate::runtime::ResidencySet>>,
    lease: Arc<StorageLease>,
    geometry: Option<Option<AffineGeometry>>,
}
// Buffer owners retain mapped storage without retaining the runtime itself.
// This avoids a pending-command -> tensor -> runtime ownership cycle.
struct StorageLease {
    _buffers: Vec<Arc<crate::runtime::Buffer>>,
    _residency: Option<Arc<crate::runtime::ResidencySet>>,
}
#[derive(Clone, Copy)]
struct AffineGeometry {
    experts: usize,
    hidden: usize,
    intermediate: usize,
    metadata: ScalarType,
}
fn affine_geometry(slots: &[&Array]) -> Option<AffineGeometry> {
    if slots.is_empty() || slots.len() % 9 != 0 {
        return None;
    }
    let first = slots.first()?;
    if first.rank() != 2 {
        return None;
    }
    let hidden = (first.dim(1) as usize).checked_mul(match first.dtype() {
        Dtype::Uint32 => 16,
        Dtype::Uint8 => 4,
        _ => return None,
    })?;
    let intermediate = first.dim(0) as usize;
    let metadata = match slots[1].dtype() {
        Dtype::Float32 => ScalarType::F32,
        Dtype::Bfloat16 => ScalarType::BF16,
        _ => return None,
    };
    if hidden == 0 || intermediate == 0 || hidden % 128 != 0 || intermediate % 128 != 0 {
        return None;
    }
    for expert in slots.chunks_exact(9) {
        for (p, (n, k)) in [
            (intermediate, hidden),
            (intermediate, hidden),
            (hidden, intermediate),
        ]
        .into_iter()
        .enumerate()
        {
            let w = expert[p * 3];
            let scale = expert[p * 3 + 1];
            let bias = expert[p * 3 + 2];
            let packed = match w.dtype() {
                Dtype::Uint32 => 16,
                Dtype::Uint8 => 4,
                _ => return None,
            };
            if w.shape() != [n as i32, (k / packed) as i32]
                || scale.shape() != [n as i32, (k / 128) as i32]
                || bias.shape() != scale.shape()
                || scale.dtype() != dtype(metadata)
                || bias.dtype() != dtype(metadata)
            {
                return None;
            }
            for a in [w, scale, bias] {
                let (buffer, layout) = a.t.buffer_and_layout();
                let end = layout
                    .offset
                    .checked_add(a.size())?
                    .checked_mul(a.dtype().size_of())?;
                if !layout.is_contiguous() || end > buffer.length() {
                    return None;
                }
            }
        }
    }
    Some(AffineGeometry {
        experts: slots.len() / 9,
        hidden,
        intermediate,
        metadata,
    })
}
impl MetalDevice {
    /// Adapter for an existing checked address table. All referenced allocations
    /// must remain live through dispatch; `owners` or the resident binding owner
    /// must retain their leases. New bindings should use `resources` instead.
    ///
    /// # Safety
    /// Every reachable table entry must be a valid address with the payload and
    /// extent required by the dispatched operation, on this runtime.
    pub unsafe fn bind_indirect_resources(
        &self,
        table: Array,
        owners: Vec<Array>,
        residency: Option<Arc<crate::runtime::ResidencySet>>,
    ) -> Result<IndirectResources> {
        if table.dtype() != Dtype::Int64
            || table.rank() != 1
            || !Arc::ptr_eq(table.t.runtime(), &self.runtime)
            || owners
                .iter()
                .any(|a| !Arc::ptr_eq(a.t.runtime(), &self.runtime))
        {
            return Err(Error::msg(
                "indirect resources: invalid table or foreign context",
            ));
        }
        let lease = Arc::new(StorageLease {
            _buffers: owners.iter().map(|a| a.t.buffer().clone()).collect(),
            _residency: residency.clone(),
        });
        Ok(IndirectResources {
            runtime: self.runtime.clone(),
            table,
            owners,
            residency,
            lease,
            geometry: None,
        })
    }
    /// Bind stable tensors to an already committed resource set. Slot order
    /// belongs to the operation contract; the provider owns address encoding.
    pub fn resources_with_residency(
        &self,
        slots: &[&Array],
        residency: Option<Arc<crate::runtime::ResidencySet>>,
    ) -> Result<IndirectResources> {
        if residency
            .as_ref()
            .is_some_and(|r| slots.iter().any(|a| !r.contains(a.backing_buffer_id())))
        {
            return Err(Error::msg(
                "indirect resources are absent from residency set",
            ));
        }
        let table = self.indirect_table(slots)?;
        let lease = Arc::new(StorageLease {
            _buffers: slots.iter().map(|a| a.t.buffer().clone()).collect(),
            _residency: residency.clone(),
        });
        Ok(IndirectResources {
            runtime: self.runtime.clone(),
            table,
            owners: slots.iter().map(|a| (*a).clone()).collect(),
            residency,
            lease,
            geometry: Some(affine_geometry(slots)),
        })
    }
    /// Encode opaque storage addresses in the supplied order. Only binding code
    /// uses this compatibility adapter; model graphs never receive the table.
    pub fn indirect_table(&self, slots: &[&Array]) -> Result<Array> {
        if slots.is_empty()
            || slots.len() > i32::MAX as usize
            || slots
                .iter()
                .any(|a| !Arc::ptr_eq(a.t.runtime(), &self.runtime))
        {
            return Err(Error::msg("indirect table: empty or foreign resources"));
        }
        let addresses = slots
            .iter()
            .map(|a| {
                a.gpu_address()
                    .map(|x| x as i64)
                    .ok_or_else(|| Error::msg("indirect storage has no device address"))
            })
            .collect::<Result<Vec<_>>>()?;
        let bytes = addresses
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect::<Vec<_>>();
        self.pull(
            HostTensor {
                bytes: &bytes,
                shape: &[addresses.len() as i32],
                dtype: ScalarType::I64,
            }
            .into(),
            ImportOptions::default(),
        )
    }
}
impl IndirectDevice for MetalDevice {
    type Resources = IndirectResources;
    fn resources(&self, slots: &[&Array]) -> Result<IndirectResources> {
        self.resources_with_residency(slots, None)
    }
    fn dispatch_indirect(
        &self,
        handle: &KernelHandle,
        inputs: &[&Array],
        resources: &IndirectResources,
        stream: &Stream,
    ) -> Result<Option<Dispatch<Outputs<Array>, Completion>>> {
        if !Arc::ptr_eq(&resources.runtime, &self.runtime) {
            return Err(Error::msg("indirect dispatch: foreign resource context"));
        }
        let output = self.dispatch_resources(handle, inputs, stream, Some(resources))?;
        if let Some(output) = &output {
            self.runtime
                .commands
                .retain_lease(output.completion.id, resources.lease.clone());
        }
        Ok(output)
    }
}
const F: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::F32],
    min_rank: 1,
    max_rank: None,
    layout: LayoutSupport::RowContiguous,
    alignment: 4,
};
const X: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::F32, ScalarType::F16],
    ..F
};
const I: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::I32],
    ..F
};
const H: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::F16],
    ..F
};
pub(super) fn capabilities() -> Vec<KernelCapability> {
    use IndirectAffineKind as K;
    let cap = |kind, inputs, write: &'static [usize]| KernelCapability {
        kind: KernelKind::IndirectAffine(kind),
        law: NumericalLaw::IndirectAffineV1,
        inputs,
        variadic: false,
        shape: ShapeRelation::TensorOperation,
        encoding: None,
        workspace: if write.is_empty() {
            Workspace::ProviderManaged
        } else {
            Workspace::CallerOutputs { inputs: write }
        },
    };
    let mut v = vec![
        cap(K::RouteMap, &[I], &[]),
        cap(K::Pair, &[X, I, F], &[]),
        cap(K::Gather(false), &[X, I], &[]),
        cap(K::Gather(true), &[X, I], &[]),
        cap(K::DownSum8(false), &[X, I], &[]),
        cap(K::DownSum8(true), &[X, I, F], &[]),
        cap(
            K::DownSum8SharedProjection,
            &[
                X,
                I,
                F,
                TensorConstraint {
                    dtypes: &[ScalarType::U32],
                    ..F
                },
                F,
                F,
            ],
            &[],
        ),
        cap(K::Sum8, &[F], &[]),
    ];
    if crate::runtime::nax_available() {
        v.extend([
            cap(K::GroupedProjection(false), &[F, I, I, I, I], &[]),
            cap(K::GroupedProjection(true), &[F, I, I, I, I, F], &[5]),
            cap(K::GroupedPair(false), &[F, I, I, I, I, F], &[]),
            cap(K::GroupedPair(true), &[F, I, I, I, I, F, H], &[6]),
            cap(K::GroupedSwiGlu(false), &[F, F, F], &[]),
            cap(K::GroupedSwiGlu(true), &[F, F, F, H], &[3]),
            cap(K::GroupedDown(false), &[H, I, I, I, I], &[]),
            cap(K::GroupedDown(true), &[H, I, I, I, I, F], &[5]),
        ]);
    }
    v
}
pub(super) fn without_resources(
    k: &Kernel,
    a: &[&Array],
    s: &Stream,
) -> Result<Option<Outputs<Array>>> {
    let Kernel::IndirectAffine(op) = k else {
        return Ok(None);
    };
    invoke(op, a, s, None)
}
pub(super) fn invoke(
    op: &IndirectAffine,
    a: &[&Array],
    s: &Stream,
    r: Option<&IndirectResources>,
) -> Result<Option<Outputs<Array>>> {
    use IndirectAffine as K;
    if op.requires_resources() && r.is_none() {
        return Err(Error::msg("indirect operation requires resources"));
    }
    if op.requires_resources() {
        if let Some(geometry) = r.and_then(|r| r.geometry) {
            let Some(g) = geometry else {
                return Err(Error::msg(
                    "indirect affine resources have incompatible payload geometry",
                ));
            };
            let valid = match *op {
                K::Pair {
                    intermediate,
                    metadata,
                    ..
                } => {
                    intermediate == g.intermediate
                        && metadata == g.metadata
                        && a[0].dim(-1) as usize == g.hidden
                }
                K::Gather {
                    hidden, metadata, ..
                }
                | K::DownSum8SharedProjection { hidden, metadata }
                | K::DownSum8 {
                    hidden, metadata, ..
                } => {
                    hidden == g.hidden
                        && metadata == g.metadata
                        && a[0].dim(-1) as usize == g.intermediate
                }
                K::GroupedProjection {
                    experts,
                    intermediate,
                    metadata,
                    ..
                }
                | K::GroupedPair {
                    experts,
                    intermediate,
                    metadata,
                    ..
                } => {
                    experts == g.experts
                        && intermediate == g.intermediate
                        && metadata == g.metadata
                        && a[0].dim(-1) as usize == g.hidden
                }
                K::GroupedDown {
                    experts,
                    hidden,
                    metadata,
                    ..
                } => {
                    experts == g.experts
                        && hidden == g.hidden
                        && metadata == g.metadata
                        && a[0].dim(-1) as usize == g.intermediate
                }
                _ => true,
            };
            if !valid {
                return Err(Error::msg(
                    "indirect affine invocation differs from resource geometry",
                ));
            }
        }
    }
    let keepalive = r
        .filter(|r| r.residency.is_none())
        .map(|r| r.owners.iter().collect::<Vec<_>>())
        .unwrap_or_default();
    let keep = keepalive.iter().map(|a| &a.t).collect::<Vec<_>>();
    let residency = r.and_then(|r| r.residency.as_deref());
    let table = || &r.expect("checked resource binding").table;
    let meta = |d| match d {
        ScalarType::BF16 => crate::jit::DType::BF16,
        ScalarType::F32 => crate::jit::DType::F32,
        _ => unreachable!("validated metadata"),
    };
    let out = match *op {
        K::RouteMap {
            rows,
            top_k,
            experts,
            tile,
        } => {
            return Ok(
                raw::moe_expert_route_map(a[0], rows, top_k, experts, tile, s)
                    .map(|(a, b, c, d)| Outputs::Many(vec![a, b, c, d])),
            );
        }
        K::Pair {
            intermediate,
            top_k,
            limit,
            metadata,
        } => raw::moe_pair_swiglu_addr(
            table(),
            a[0],
            a[1],
            a[2],
            &keepalive,
            intermediate,
            top_k,
            f32::from_bits(limit),
            dtype(metadata),
            residency,
            s,
        ),
        K::Gather {
            hidden,
            ds4,
            metadata,
        } => {
            let f = if ds4 {
                raw::moe_gather_qmv_addr_ds4
            } else {
                raw::moe_gather_qmv_addr
            };
            f(
                table(),
                a[0],
                a[1],
                &keepalive,
                hidden,
                true,
                dtype(metadata),
                residency,
                s,
            )
        }
        K::DownSum8 {
            hidden,
            shared,
            metadata,
        } => raw::moe_gather_qmv_sum8_addr_ds4(
            table(),
            a[0],
            a[1],
            &keepalive,
            if shared { Some(a[2]) } else { None },
            hidden,
            dtype(metadata),
            residency,
            s,
        ),
        K::DownSum8SharedProjection { hidden, metadata } => {
            raw::moe_gather_qmv_sum8_addr_add_shared_down(
                table(),
                a[0],
                a[1],
                &keepalive,
                a[2],
                a[3],
                a[4],
                a[5],
                hidden,
                dtype(metadata),
                residency,
                s,
            )
        }
        K::Sum8 { rows, hidden } => raw::moe_sum8(a[0], rows, hidden, s),
        K::GroupedProjection {
            rows,
            top_k,
            experts,
            intermediate,
            up,
            vector,
            metadata,
            workspace,
        } => Some(Array::new(
            crate::jit::glm53_moe_q2_mm_grouped_ds4_prefill_with_output(
                a[0].t.device(),
                &table().t,
                &a[0].t,
                &a[1].t,
                &a[2].t,
                &a[3].t,
                &a[4].t,
                &keep,
                residency,
                meta(metadata),
                rows,
                top_k,
                experts,
                intermediate,
                up,
                vector,
                if workspace { Some(&a[5].t) } else { None },
            )?,
        )),
        K::GroupedPair {
            rows,
            top_k,
            experts,
            intermediate,
            limit,
            metadata,
            workspace,
        } => Some(Array::new(
            crate::jit::glm53_moe_q2_pair_grouped_ds4_prefill_with_output(
                a[0].t.device(),
                &table().t,
                &a[0].t,
                &a[1].t,
                &a[2].t,
                &a[3].t,
                &a[4].t,
                &a[5].t,
                &keep,
                residency,
                meta(metadata),
                rows,
                top_k,
                experts,
                intermediate,
                f32::from_bits(limit),
                if workspace { Some(&a[6].t) } else { None },
            )?,
        )),
        K::GroupedSwiGlu {
            rows,
            intermediate,
            limit,
            workspace,
        } => Some(Array::new(
            crate::jit::glm53_moe_grouped_swiglu_weight_prefill_with_output(
                a[0].t.device(),
                &a[0].t,
                &a[1].t,
                &a[2].t,
                rows,
                intermediate,
                f32::from_bits(limit),
                if workspace { Some(&a[3].t) } else { None },
            )?,
        )),
        K::GroupedDown {
            rows,
            top_k,
            experts,
            hidden,
            vector,
            tile,
            metadata,
            workspace,
        } => Some(Array::new(
            crate::jit::glm53_moe_q2_down_grouped_ds4_prefill_with_output(
                a[0].t.device(),
                &table().t,
                &a[0].t,
                &a[1].t,
                &a[2].t,
                &a[3].t,
                &a[4].t,
                &keep,
                residency,
                meta(metadata),
                rows,
                top_k,
                experts,
                hidden,
                vector,
                tile,
                if workspace { Some(&a[5].t) } else { None },
            )?,
        )),
    };
    Ok(out.map(Outputs::One))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn indirect_resources_reject_incompatible_payload_before_dispatch() {
        let d = MetalDevice::system_default();
        let s = d.stream();
        let x = Array::from_slice(&vec![0f32; 512], &[1, 512]);
        let selected = Array::from_slice(&[0i32], &[1]);
        let weights = Array::from_slice(&[1f32], &[1]);
        let resources = d.resources(&[&x]).unwrap();
        let k = d
            .prepare(Kernel::IndirectAffine(IndirectAffine::Pair {
                intermediate: 512,
                top_k: 1,
                limit: 10f32.to_bits(),
                metadata: ScalarType::F32,
            }))
            .unwrap()
            .unwrap();
        assert!(
            d.dispatch_indirect(&k, &[&x, &selected, &weights], &resources, &s)
                .is_err()
        );
        assert!(d.dispatch(&k, &[&x, &selected, &weights], &s).is_err());
    }
    #[test]
    fn completion_leases_survive_dropping_the_binding_and_handle_late_registration() {
        use std::sync::atomic::{AtomicBool, Ordering};
        struct Owner(Arc<AtomicBool>);
        impl Drop for Owner {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let d = MetalDevice::system_default();
        let s = d.stream();
        s.runtime().commands.flush_and_wait().unwrap();
        let x = Array::from_slice(&[1.25f32, 2.5, 3.75], &[3]);
        let k = d
            .prepare(Kernel::Tensor(TensorOperation::Binary(Binary::Add)))
            .unwrap()
            .unwrap();
        let out = d.dispatch(&k, &[&x, &x], &s).unwrap().unwrap();
        assert!(out.completion.id > 0);
        let dropped = Arc::new(AtomicBool::new(false));
        let owner = Arc::new(Owner(dropped.clone()));
        s.runtime()
            .commands
            .retain_lease(out.completion.id, owner.clone());
        drop(owner);
        d.wait(&out.completion).unwrap();
        assert!(dropped.load(Ordering::SeqCst));
        let late = Arc::new(AtomicBool::new(false));
        s.runtime()
            .commands
            .retain_lease(out.completion.id, Arc::new(Owner(late.clone())));
        assert!(late.load(Ordering::SeqCst));
    }
}
