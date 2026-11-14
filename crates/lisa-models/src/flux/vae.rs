//! FLUX autoencoder decoder, preserving the original normalization reductions,
//! NHWC convolution gather order and 1024-row convolution chunk bound.
use crate::{
    graph::Value,
    media_attention,
    program::{Build, Program, Schema},
};
use lisa_backend::{Binary as B, Kernel, ScalarType as S, TensorOperation as T, Unary as U};
use std::collections::{BTreeSet, HashMap};
#[derive(Clone, Copy)]
struct Image {
    value: Value,
    h: i32,
    w: i32,
    c: i32,
}
impl Image {
    fn shape(self) -> [i32; 4] {
        [1, self.h, self.w, self.c]
    }
}
struct Decoder<'a> {
    p: Build<'a>,
    dtype: S,
    quantized: &'a BTreeSet<String>,
    // Gather geometry is independent of weights; share immutable index constants.
    indices: HashMap<(i32, i32, i32), Value>,
}
impl Decoder<'_> {
    fn norm(&mut self, x: Image, name: &str) -> anyhow::Result<Image> {
        anyhow::ensure!(x.c % 32 == 0, "VAE group norm requires 32 channel groups");
        let p = &mut self.p;
        let flat = p.cast(x.value, S::F32);
        let flat = p.reshape(flat, &[1, x.h * x.w, 32, x.c / 32]);
        let flat = p.transpose(flat, &[0, 2, 1, 3]);
        let flat = p.contiguous(flat);
        let flat = p.reshape(flat, &[1, 32, x.h * x.w * x.c / 32]);
        let mean = p.op(
            T::Mean {
                axis: -1,
                keep_dims: true,
            },
            vec![flat],
        );
        let centered = p.binary(B::Subtract, flat, mean);
        let squared = p.binary(B::Multiply, centered, centered);
        let variance = p.op(
            T::Mean {
                axis: -1,
                keep_dims: true,
            },
            vec![squared],
        );
        let eps = p.scalar(1e-6, S::F32);
        let variance = p.binary(B::Add, variance, eps);
        let inv = p.unary(U::Rsqrt, variance);
        let y = p.binary(B::Multiply, centered, inv);
        let y = p.reshape(y, &[1, 32, x.h * x.w, x.c / 32]);
        let y = p.transpose(y, &[0, 2, 1, 3]);
        let y = p.contiguous(y);
        let y = p.reshape(y, &x.shape());
        let weight = p.weight(&format!("{name}.weight"));
        let weight = p.cast(weight, S::F32);
        let bias = p.weight(&format!("{name}.bias"));
        let bias = p.cast(bias, S::F32);
        let y = p.binary(B::Multiply, y, weight);
        let y = p.binary(B::Add, y, bias);
        let value = p.cast(y, self.dtype);
        Ok(Image { value, ..x })
    }
    fn conv(&mut self, x: Image, name: &str) -> anyhow::Result<Image> {
        let shape = self
            .p
            .schema
            .get(&format!("{name}.weight"))
            .ok_or_else(|| anyhow::anyhow!("missing VAE convolution {name}"))?;
        anyhow::ensure!(
            shape.len() == 4
                && shape[3] == x.c
                && shape[1] == shape[2]
                && [1, 3].contains(&shape[1]),
            "unsupported VAE convolution {name}"
        );
        let (kernel, output) = (shape[1], shape[0]);
        let rows = x.h * x.w;
        let p = &mut self.p;
        let weight = p.weight(&format!("{name}.weight"));
        let bias = p.weight(&format!("{name}.bias"));
        let flat = p.contiguous(x.value);
        let flat = p.reshape(flat, &[rows, x.c]);
        let wt = p.contiguous(weight);
        let wt = p.reshape(wt, &[output, kernel * kernel * x.c]);
        let wt = p.transpose(wt, &[1, 0]);
        let value = if kernel == 1 {
            let y = p.op(T::Matmul, vec![flat, wt]);
            let y = p.binary(B::Add, y, bias);
            p.reshape(y, &[1, x.h, x.w, output])
        } else {
            let zero = p.graph.constant_f32(&vec![0.; x.c as usize], &[1, x.c]);
            let zero = p.cast(zero, self.dtype);
            let source = p.concat(vec![flat, zero], 0);
            let mut pieces = Vec::new();
            for start in (0..rows).step_by(1024) {
                let count = (rows - start).min(1024);
                let index = if let Some(&ids) = self.indices.get(&(x.h, x.w, start)) {
                    ids
                } else {
                    let mut ids = Vec::with_capacity((count * 9) as usize);
                    for row in start..start + count {
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let y = row / x.w + ky - 1;
                                let xx = row % x.w + kx - 1;
                                ids.push(if y < 0 || y >= x.h || xx < 0 || xx >= x.w {
                                    rows
                                } else {
                                    y * x.w + xx
                                });
                            }
                        }
                    }
                    let ids = p.graph.constant_i32(&ids, &[count * 9]);
                    self.indices.insert((x.h, x.w, start), ids);
                    ids
                };
                let patches = p.op(T::Take { axis: 0 }, vec![source, index]);
                let patches = p.reshape(patches, &[count, 9 * x.c]);
                let y = p.op(T::Matmul, vec![patches, wt]);
                pieces.push(p.binary(B::Add, y, bias));
            }
            let y = p.concat(pieces, 0);
            p.reshape(y, &[1, x.h, x.w, output])
        };
        Ok(Image {
            value,
            c: output,
            ..x
        })
    }
    fn silu(&mut self, x: Image) -> Image {
        let sigmoid = self.p.unary(U::Sigmoid, x.value);
        let value = self.p.binary(B::Multiply, x.value, sigmoid);
        Image { value, ..x }
    }
    fn resnet(&mut self, x: Image, name: &str) -> anyhow::Result<Image> {
        let n = self.norm(x, &format!("{name}.norm1"))?;
        let n = self.silu(n);
        let h = self.conv(n, &format!("{name}.conv1"))?;
        let n = self.norm(h, &format!("{name}.norm2"))?;
        let n = self.silu(n);
        let h = self.conv(n, &format!("{name}.conv2"))?;
        let skip = if self
            .p
            .schema
            .contains_key(&format!("{name}.conv_shortcut.weight"))
        {
            self.conv(x, &format!("{name}.conv_shortcut"))?
        } else {
            x
        };
        anyhow::ensure!(h.shape() == skip.shape(), "VAE residual shape mismatch");
        let value = self.p.binary(B::Add, h.value, skip.value);
        Ok(Image { value, ..h })
    }
    fn upsample(&mut self, x: Image) -> Image {
        let y = self
            .p
            .graph
            .constant_i32(&(0..x.h * 2).map(|i| i / 2).collect::<Vec<_>>(), &[x.h * 2]);
        let w = self
            .p
            .graph
            .constant_i32(&(0..x.w * 2).map(|i| i / 2).collect::<Vec<_>>(), &[x.w * 2]);
        let y = self.p.op(T::Take { axis: 1 }, vec![x.value, y]);
        let y = self.p.op(T::Take { axis: 2 }, vec![y, w]);
        let value = self.p.reshape(y, &[1, x.h * 2, x.w * 2, x.c]);
        Image {
            value,
            h: x.h * 2,
            w: x.w * 2,
            ..x
        }
    }
    fn linear(&mut self, x: Image, name: &str) -> anyhow::Result<Image> {
        let p = &mut self.p;
        let weight_name = format!("{name}.weight");
        let shape = p
            .schema
            .get(&weight_name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing {weight_name}"))?;
        anyhow::ensure!(shape.len() == 2, "invalid VAE linear {name}");
        let out = shape[0];
        let weight = p.weight(&weight_name);
        let y = if self.quantized.contains(&weight_name) {
            let scales_name = format!("{name}.scales");
            let scales = p
                .schema
                .get(&scales_name)
                .ok_or_else(|| anyhow::anyhow!("missing {scales_name}"))?;
            anyhow::ensure!(
                scales.len() == 2 && shape[1] * 8 == scales[1] * 64 && x.c == scales[1] * 64,
                "VAE requires affine 4-bit/group-64 weights at {name}"
            );
            let scales = p.weight(&scales_name);
            let biases = p.weight(&format!("{name}.biases"));
            p.graph.one(
                Kernel::AffineLinear {
                    group_size: 64,
                    bits: 4,
                },
                vec![x.value, weight, scales, biases],
            )
        } else {
            anyhow::ensure!(shape[1] == x.c, "invalid VAE dense projection {name}");
            let wt = p.transpose(weight, &[1, 0]);
            let wt = p.contiguous(wt);
            let flat = p.reshape(x.value, &[-1, x.c]);
            let y = p.op(T::Matmul, vec![flat, wt]);
            p.reshape(y, &[1, x.h, x.w, out])
        };
        let bias = format!("{name}.bias");
        let value = if p.schema.contains_key(&bias) {
            let bias = p.weight(&bias);
            p.binary(B::Add, y, bias)
        } else {
            y
        };
        Ok(Image { value, c: out, ..x })
    }
}
/// Packed latent input is [1, 128, height, width]. All model weights retain
/// their original payload and dtype; only the explicit graph casts may change it.
pub fn decoder(
    schema: &Schema,
    quantized: &BTreeSet<String>,
    height: i32,
    width: i32,
    dtype: S,
) -> anyhow::Result<Program> {
    anyhow::ensure!(
        height > 0 && width > 0 && height <= 64 && width <= 64 && matches!(dtype, S::BF16 | S::F32),
        "unsupported VAE geometry or dtype"
    );
    let mut d = Decoder {
        p: Build::new(1, schema),
        dtype,
        quantized,
        indices: HashMap::new(),
    };
    let p = &mut d.p;
    let mean = p.weight("bn.running_mean");
    let mean = p.cast(mean, S::F32);
    let mean = p.reshape(mean, &[1, 128, 1, 1]);
    let variance = p.weight("bn.running_var");
    let variance = p.cast(variance, S::F32);
    let eps = p.scalar(1e-4, S::F32);
    let variance = p.binary(B::Add, variance, eps);
    let std = p.unary(U::Sqrt, variance);
    let std = p.reshape(std, &[1, 128, 1, 1]);
    let x = p.cast(Value::Input(0), S::F32);
    let x = p.binary(B::Multiply, x, std);
    let x = p.binary(B::Add, x, mean);
    let x = p.reshape(x, &[1, 32, 2, 2, height, width]);
    let x = p.transpose(x, &[0, 1, 4, 2, 5, 3]);
    let x = p.contiguous(x);
    let x = p.reshape(x, &[1, 32, height * 2, width * 2]);
    let x = p.transpose(x, &[0, 2, 3, 1]);
    let value = p.cast(x, dtype);
    let x = Image {
        value,
        h: height * 2,
        w: width * 2,
        c: 32,
    };
    let x = d.conv(x, "post_quant_conv")?;
    let mut x = d.conv(x, "decoder.conv_in")?;
    x = d.resnet(x, "decoder.mid_block.resnets.0")?;
    let name = "decoder.mid_block.attentions.0";
    let n = d.norm(x, &format!("{name}.group_norm"))?;
    let mut qkv = Vec::new();
    for key in ["to_q", "to_k", "to_v"] {
        let v = d.linear(n, &format!("{name}.{key}"))?;
        qkv.push(d.p.reshape(v.value, &[1, 1, x.h * x.w, x.c]));
    }
    let shape = [1, 1, x.h * x.w, x.c];
    let geometry = media_attention::Geometry {
        q: shape,
        k: shape,
        v: shape,
        output_dtype: dtype,
        scale: (1.0 / (x.c as f32).sqrt()).to_bits(),
        mask: false,
    };
    let attn = media_attention::append(&mut d.p, &geometry, qkv[0], qkv[1], qkv[2], None)?;
    let value = d.p.reshape(attn, &x.shape());
    let out = d.linear(Image { value, ..x }, &format!("{name}.to_out"))?;
    x.value = d.p.binary(B::Add, x.value, out.value);
    x = d.resnet(x, "decoder.mid_block.resnets.1")?;
    for block in 0..4 {
        for layer in 0..3 {
            x = d.resnet(x, &format!("decoder.up_blocks.{block}.resnets.{layer}"))?;
        }
        if block < 3 {
            let up = d.upsample(x);
            x = d.conv(up, &format!("decoder.up_blocks.{block}.upsamplers.0.conv"))?;
        }
    }
    let x = d.norm(x, "decoder.conv_norm_out")?;
    let x = d.silu(x);
    let x = d.conv(x, "decoder.conv_out")?;
    d.p.finish(vec![x.value])
}
