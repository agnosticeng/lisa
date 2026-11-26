//! Checkpoint-defined rotary coefficients, independent of device kernels.
/// The same host-side powf/reciprocal and YaRN operation order as the reference.
pub fn rope_frequencies(compressed: bool) -> &'static [f32; 32] {
    static FREQUENCIES: std::sync::OnceLock<[[f32; 32]; 2]> = std::sync::OnceLock::new();
    &FREQUENCIES.get_or_init(|| {
        let mut result = [[0f32; 32]; 2];
        for (kind, frequencies) in result.iter_mut().enumerate() {
            let base = if kind == 0 { 10000f32 } else { 160000f32 };
            let low = (64.0 * (65536.0 / (32.0 * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * (base as f64).ln()))
            .floor() as f32;
            let high = (64.0 * (65536.0 / (2.0 * std::f64::consts::PI)).ln()
                / (2.0 * (base as f64).ln()))
            .ceil() as f32;
            for (i, frequency) in frequencies.iter_mut().enumerate() {
                let denominator = base.powf(i as f32 / 32.0);
                let mut f = 1.0 / denominator;
                if kind != 0 {
                    let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                    let smooth = 1.0 - ramp;
                    let interpolate = (f / 16.0) * (1.0 - smooth);
                    let extrapolate = f * smooth;
                    f = interpolate + extrapolate;
                }
                *frequency = f;
            }
        }
        result
    })[usize::from(compressed)]
}

#[cfg(test)]
mod tests {
    #[test]
    fn coefficients_preserve_pre_extraction_bits() {
        let expected: [[u32; 32]; 2] = [
            [
                1065353216, 1061157137, 1058010522, 1054337179, 1050798235, 1047712803, 1043732615,
                1040747895, 1036831949, 1033475085, 1030116802, 1026341397, 1023510243, 1019364175,
                1016180025, 1012562930, 1008981770, 1005959600, 1001931932, 998911607, 995049016,
                991652109, 988353832, 984533390, 981668463, 977573004, 974350870, 970790693,
                967166816, 964208661, 960132947, 957076594,
            ],
            [
                1065353216, 1060112954, 1056054303, 1051098369, 1046804782, 1042117749, 1037601972,
                1033169251, 1028443341, 1024251131, 1019326489, 1015361742, 1010249151, 1006366088,
                1001209182, 997081655, 991197921, 985076960, 979287478, 973418702, 966505391,
                959469944, 952012877, 943627103, 933484300, 917783504, 913828774, 908753896,
                904558591, 899759068, 895336240, 890797133,
            ],
        ];
        for (i, compressed) in [false, true].into_iter().enumerate() {
            assert_eq!(
                super::rope_frequencies(compressed).map(f32::to_bits),
                expected[i]
            );
        }
    }
}

pub mod moe;

pub mod projection;

pub mod config;

pub mod engram;

pub mod weights;

pub mod hyper;

pub mod state;

pub mod attention;

pub mod tower;
