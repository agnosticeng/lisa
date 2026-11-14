#![cfg(test)]
use super::*;
use crate::bindings::deepseek41_moe::tests::{counts, difference};
#[path = "glm53_recurrent_plan_oracle.rs"]
mod oracle;
fn values(shape: &[i32], seed: usize) -> Array {
    Array::from_slice(
        &(0..shape.iter().map(|&x| x as usize).product::<usize>())
            .map(|i| (((i * 17 + seed) % 127) as f32 - 63.) / 4096.)
            .collect::<Vec<_>>(),
        shape,
    )
}
fn pair() -> anyhow::Result<(Kda, oracle::Kda)> {
    let n = 64 * 128;
    let p = |shape: &[i32], seed| -> anyhow::Result<Projection> {
        Ok(values(shape, seed).as_dtype(Dtype::Bfloat16)?.into())
    };
    let k = Kda {
        context: crate::metal_context::execution_context(),
        q_w: p(&[n, 128], 1)?,
        k_w: p(&[n, 128], 2)?,
        v_w: p(&[n, 128], 3)?,
        o_proj: p(&[128, n], 4)?,
        f_a_w: p(&[128, 128], 5)?,
        f_b_w: p(&[n, 128], 6)?,
        g_a_w: p(&[128, 128], 7)?,
        g_b_w: p(&[n, 128], 8)?,
        b_w: p(&[64, 128], 9)?,
        conv_q: values(&[n, 4], 11),
        conv_k: values(&[n, 4], 12),
        conv_v: values(&[n, 4], 13),
        dt_bias: values(&[64, 128], 14),
        a_log: values(&[64], 15),
        o_norm_w: values(&[128], 16),
        heads: 64,
        head_dim: 128,
        conv_k_sz: 4,
        lower_bound: -5.,
        norm_eps: 1e-6,
    };
    let old = oracle::Kda {
        q_w: k.q_w.clone().into(),
        k_w: k.k_w.clone().into(),
        v_w: k.v_w.clone().into(),
        o_proj: k.o_proj.clone().into(),
        f_a_w: k.f_a_w.clone().into(),
        f_b_w: k.f_b_w.clone().into(),
        g_a_w: k.g_a_w.clone().into(),
        g_b_w: k.g_b_w.clone().into(),
        b_w: k.b_w.clone().into(),
        conv_q: k.conv_q.clone(),
        conv_k: k.conv_k.clone(),
        conv_v: k.conv_v.clone(),
        dt_bias: k.dt_bias.clone(),
        a_log: k.a_log.clone(),
        o_norm_w: k.o_norm_w.clone(),
        heads: k.heads,
        head_dim: k.head_dim,
        conv_k_sz: k.conv_k_sz,
        lower_bound: k.lower_bound,
        norm_eps: k.norm_eps,
    };
    Ok((k, old))
}
#[test]
fn recurrent_plans_preserve_states_bits_and_dispatch_order() -> anyhow::Result<()> {
    let (new, old) = pair()?;
    for (b, t, prefill) in [
        (1, 1, false),
        (1, 1, true),
        (1, 2, true),
        (1, 8, true),
        (1, 9, true),
        (2, 1, false),
        (8, 1, false),
        (2, 3, false),
    ] {
        for prior in [false, true] {
            let x = values(&[b, t, 128], 27);
            let mut c = prior.then(|| values(&[b, 3, 3 * 8192], 31));
            let mut s = prior.then(|| values(&[b, 64, 128, 128], 37));
            let (mut cc, mut ss) = (c.clone(), s.clone());
            let a = counts();
            let y = if prefill {
                old.forward_prefill(&x, &mut c, &mut s, None, 0)?
            } else {
                old.forward(&x, &mut c, &mut s, None, 0)?
            };
            let z = counts();
            let yy = if prefill {
                new.forward_prefill(&x, &mut cc, &mut ss, None, 0)?
            } else {
                new.forward(&x, &mut cc, &mut ss, None, 0)?
            };
            let zz = counts();
            for (name, a, b_) in [
                ("output", &y, &yy),
                ("conv", c.as_ref().unwrap(), cc.as_ref().unwrap()),
                ("state", s.as_ref().unwrap(), ss.as_ref().unwrap()),
            ] {
                assert_eq!(a.shape(), b_.shape(), "{name} b={b} t={t} prior={prior}");
                assert!(
                    a.to_vec1::<u32>()? == b_.to_vec1::<u32>()?,
                    "{name} b={b} t={t} prior={prior}"
                );
            }
            if std::env::var_os("LISA_TRACE").is_some() {
                assert_eq!(
                    difference(&a, &z),
                    difference(&z, &zz),
                    "b={b} t={t} prior={prior}"
                );
            }
        }
    }
    Ok(())
}
#[test]
fn recurrent_plan_observations_match_legacy_capture_values() -> anyhow::Result<()> {
    let (new, old) = pair()?;
    for (b, t, prior) in [
        (1, 1, false),
        (1, 1, true),
        (1, 2, true),
        (2, 1, true),
        (2, 3, true),
    ] {
        let x = values(&[b, t, 128], 41);
        let mut c = prior.then(|| values(&[b, 3, 3 * 8192], 43));
        let mut s = prior.then(|| values(&[b, 64, 128, 128], 47));
        let (mut cc, mut ss) = (c.clone(), s.clone());
        let cap = || capture::Capture {
            layers: vec![0],
            preserve_dispatch: true,
            ..Default::default()
        };
        let (a, z) = (cap(), cap());
        old.forward(&x, &mut c, &mut s, Some(&a), 0)?;
        new.forward(&x, &mut cc, &mut ss, Some(&z), 0)?;
        let a = a.pending.lock().unwrap();
        let z = z.pending.lock().unwrap();
        let mut ak = a.keys().collect::<Vec<_>>();
        ak.sort();
        let mut zk = z.keys().collect::<Vec<_>>();
        zk.sort();
        assert_eq!(ak, zk, "capture names b={b} t={t} prior={prior}");
        for (name, value) in a.iter() {
            assert_eq!(value.shape(), z[name].shape(), "{name}");
            assert!(
                value.as_dtype(Dtype::Float32)?.to_vec1::<u32>()?
                    == z[name].as_dtype(Dtype::Float32)?.to_vec1::<u32>()?,
                "capture {name} b={b} t={t} prior={prior}"
            );
        }
    }
    Ok(())
}
#[test]
fn recurrent_plan_preserves_independent_ragged_cache_layouts() -> anyhow::Result<()> {
    let (new, old) = pair()?;
    let x = values(&[4, 1, 128], 53);
    let mut before = Vec::new();
    let mut after = Vec::new();
    for row in 0..4 {
        let mut a = GdnCache::new();
        if row & 1 != 0 {
            a.conv = Some(values(&[1, 3, 3 * 8192], 59 + row));
        }
        if row & 2 != 0 {
            a.ssm = Some(values(&[1, 1, 64, 128, 128], 67 + row));
        }
        let mut b = GdnCache::new();
        b.conv = a.conv.clone();
        b.ssm = a.ssm.clone();
        before.push(a);
        after.push(b);
    }
    let a = counts();
    let old = old.forward_ragged(&x, &mut before.iter_mut().collect::<Vec<_>>(), None, 0)?;
    let b = counts();
    let new = new.forward_ragged(&x, &mut after.iter_mut().collect::<Vec<_>>(), None, 0)?;
    let c = counts();
    assert_eq!(old.to_vec1::<u32>()?, new.to_vec1::<u32>()?);
    for (old, new) in before.iter().zip(&after) {
        for (old, new) in [(&old.conv, &new.conv), (&old.ssm, &new.ssm)] {
            let (old, new) = (old.as_ref().unwrap(), new.as_ref().unwrap());
            assert_eq!(old.shape(), new.shape());
            assert!(old.to_vec1::<u32>()? == new.to_vec1::<u32>()?);
        }
    }
    if std::env::var_os("LISA_TRACE").is_some() {
        assert_eq!(difference(&a, &b), difference(&b, &c));
    }
    Ok(())
}

pub(crate) fn tower_fixture(width: i32) -> anyhow::Result<Kda> {
    let n = 64 * 128;
    let p = |shape: &[i32], seed| -> anyhow::Result<Projection> {
        Ok(values(shape, seed).as_dtype(Dtype::Bfloat16)?.into())
    };
    let k = Kda {
        context: crate::metal_context::execution_context(),
        q_w: p(&[n, width], 1)?,
        k_w: p(&[n, width], 2)?,
        v_w: p(&[n, width], 3)?,
        o_proj: p(&[width, n], 4)?,
        f_a_w: p(&[128, width], 5)?,
        f_b_w: p(&[n, 128], 6)?,
        g_a_w: p(&[128, width], 7)?,
        g_b_w: p(&[n, 128], 8)?,
        b_w: p(&[64, width], 9)?,
        conv_q: values(&[n, 4], 11),
        conv_k: values(&[n, 4], 12),
        conv_v: values(&[n, 4], 13),
        dt_bias: values(&[64, 128], 14),
        a_log: values(&[64], 15),
        o_norm_w: values(&[128], 16),
        heads: 64,
        head_dim: 128,
        conv_k_sz: 4,
        lower_bound: -5.,
        norm_eps: 1e-6,
    };

    Ok(k)
}
