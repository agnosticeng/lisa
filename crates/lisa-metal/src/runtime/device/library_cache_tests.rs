//! Actual Metal library/function/pipeline checks; no model data is required.
use super::*;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
constant bool NEGATE [[function_constant(0)]];
constant int ADD [[function_constant(1)]];
constant short SCALE [[function_constant(2)]];
kernel void library_cache_specialized(device const int* x [[buffer(0)]],
                                     device int* y [[buffer(1)]],
                                     uint i [[thread_position_in_grid]]) {
    y[i] = (NEGATE ? -x[i] : x[i]) * int(SCALE) + ADD;
}
kernel void library_cache_plain(device const int* x [[buffer(0)]],
                               device int* y [[buffer(1)]],
                               uint i [[thread_position_in_grid]]) {
    y[i] = x[i] - 7;
}
"#;

fn run(runtime: &MetalRuntime, pipeline: &ComputePipeline) -> Vec<i32> {
    let input = runtime.buffer(16 * 4, "library_cache_input").unwrap();
    let output = runtime.buffer(16 * 4, "library_cache_output").unwrap();
    // SAFETY: fresh 64-byte lease with no GPU user; 16 i32 writes fit.
    unsafe {
        for i in 0..16 {
            *input.contents().cast::<i32>().add(i) = i as i32 - 8;
        }
    }
    {
        let guard = runtime.commands.encoder().unwrap();
        let encoder = guard.encoder();
        encoder.set_pipeline(pipeline);
        encoder.set_input(0, Some(&input), 0);
        encoder.set_output(1, Some(&output), 0);
        encoder.dispatch_threads((16, 1, 1), (16, 1, 1));
    }
    runtime.synchronize().unwrap();
    // SAFETY: synchronized above; `output` holds 16 i32 values.
    unsafe { std::slice::from_raw_parts(output.contents().cast::<i32>(), 16).to_vec() }
}

fn constants(negate: bool, add: i32, scale: i16) -> [(usize, ConstVal); 3] {
    [
        (0, ConstVal::Bool(negate)),
        (1, ConstVal::Int(add)),
        (2, ConstVal::Short(scale)),
    ]
}

#[test]
fn library_cache_specializations_and_multiple_functions_execute_distinct_values() {
    let runtime = MetalRuntime::new(4).unwrap();
    for (negate, add, scale) in [(false, 3, 2), (true, -11, -3), (false, 5, -4)] {
        let pipeline = runtime
            .compile_with_constants(
                SOURCE,
                "library_cache_specialized",
                Math::Safe,
                &constants(negate, add, scale),
            )
            .unwrap();
        let expected: Vec<i32> = (-8..8)
            .map(|x| (if negate { -x } else { x }) * i32::from(scale) + add)
            .collect();
        assert_eq!(run(&runtime, &pipeline), expected);
    }
    let plain = runtime.compile(SOURCE, "library_cache_plain").unwrap();
    assert_eq!(
        run(&runtime, &plain),
        (-8..8).map(|x| x - 7).collect::<Vec<_>>()
    );
    assert_eq!(
        runtime.libraries.lock().unwrap().stats(),
        (4, 1, 1, SOURCE.len())
    );

    // Deliberately invalid source cannot be parsed or hashed on a pipeline hit.
    // This preserves the existing fast-path/source-twin-check contract.
    let before = runtime.libraries.lock().unwrap().stats();
    let hit = runtime
        .compile(
            "not valid Metal; must remain unread on a pipeline hit",
            "library_cache_plain",
        )
        .unwrap();
    assert_eq!(run(&runtime, &hit), run(&runtime, &plain));
    assert_eq!(runtime.libraries.lock().unwrap().stats(), before);
}

#[test]
fn library_cache_source_math_nax_and_runtime_identity_remain_separate() {
    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void library_cache_math(device const int* x [[buffer(0)]],
                               device int* y [[buffer(1)]],
                               uint i [[thread_position_in_grid]]) { y[i] = x[i] + 31; }
kernel void library_cache_other_source(device const int* x [[buffer(0)]],
                                       device int* y [[buffer(1)]],
                                       uint i [[thread_position_in_grid]]) { y[i] = x[i] + 31; }
"#;
    let mut runtime = MetalRuntime::new(4).unwrap();
    for math in [
        Math::CompilerDefault,
        Math::Jit,
        Math::Safe,
        Math::SafeNoLang,
        Math::Fast,
    ] {
        let p = runtime
            .compile_with(SOURCE, "library_cache_math", math)
            .unwrap();
        assert_eq!(
            run(&runtime, &p),
            (-8..8).map(|x| x + 31).collect::<Vec<_>>()
        );
    }
    assert_eq!(runtime.libraries.lock().unwrap().stats().1, 5);
    if runtime.nax {
        // The same physical device can compile MSL 3.2 and its NAX target.
        // Non-NAX hardware still exercises both identities in the host test.
        runtime.nax = false;
        let p = runtime.compile(SOURCE, "library_cache_math").unwrap();
        assert_eq!(
            run(&runtime, &p),
            (-8..8).map(|x| x + 31).collect::<Vec<_>>()
        );
        assert_eq!(runtime.libraries.lock().unwrap().stats().1, 6);
        runtime.nax = true;
    }
    let before = runtime.libraries.lock().unwrap().stats().1;
    let changed_source = SOURCE.replace("+ 31", "+ 47");
    let changed = runtime
        .compile(&changed_source, "library_cache_other_source")
        .unwrap();
    assert_eq!(
        run(&runtime, &changed),
        (-8..8).map(|x| x + 47).collect::<Vec<_>>()
    );
    assert_eq!(runtime.libraries.lock().unwrap().stats().1, before + 1);

    let other = MetalRuntime::new(4).unwrap();
    let p = other.compile(SOURCE, "library_cache_math").unwrap();
    assert_eq!(run(&other, &p), (-8..8).map(|x| x + 31).collect::<Vec<_>>());
    assert_eq!(
        other.libraries.lock().unwrap().stats(),
        (1, 1, 1, SOURCE.len())
    );
}

#[test]
fn library_cache_concurrent_constant_requests_create_one_library() {
    let runtime = Arc::new(MetalRuntime::new(4).unwrap());
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let handles: Vec<_> = (0..8)
        .map(|i| {
            let runtime = runtime.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                runtime
                    .compile_with_constants(
                        SOURCE,
                        "library_cache_specialized",
                        Math::Safe,
                        &constants(false, i, 2),
                    )
                    .unwrap()
            })
        })
        .collect();
    for (i, handle) in handles.into_iter().enumerate() {
        let p = handle.join().unwrap();
        assert_eq!(
            run(&runtime, &p),
            (-8..8).map(|x| x * 2 + i as i32).collect::<Vec<_>>()
        );
    }
    assert_eq!(
        runtime.libraries.lock().unwrap().stats(),
        (8, 1, 1, SOURCE.len())
    );
}

#[test]
fn library_cache_invalid_source_retries_and_function_errors_reuse_valid_library() {
    const GOOD: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void library_cache_after_error(device const int* x [[buffer(0)]],
                                     device int* y [[buffer(1)]],
                                     uint i [[thread_position_in_grid]]) { y[i] = x[i] + 9; }
"#;
    let runtime = MetalRuntime::new(4).unwrap();
    for _ in 0..2 {
        let e = runtime
            .compile("this is not valid Metal", "library_cache_after_error")
            .err()
            .unwrap();
        assert!(
            e.to_string()
                .starts_with("library `library_cache_after_error`:")
        );
    }
    assert_eq!(runtime.libraries.lock().unwrap().stats(), (2, 2, 0, 0));
    for _ in 0..2 {
        let e = runtime
            .compile(GOOD, "library_cache_missing")
            .err()
            .unwrap();
        assert_eq!(e.to_string(), "function `library_cache_missing`");
    }
    assert_eq!(
        runtime.libraries.lock().unwrap().stats(),
        (4, 3, 1, GOOD.len())
    );
    let p = runtime.compile(GOOD, "library_cache_after_error").unwrap();
    assert_eq!(
        run(&runtime, &p),
        (-8..8).map(|x| x + 9).collect::<Vec<_>>()
    );
    assert_eq!(
        runtime.libraries.lock().unwrap().stats(),
        (5, 3, 1, GOOD.len())
    );
}
