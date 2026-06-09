use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

// The library target is named "_trimat" (see [lib] name in Cargo.toml), so the
// rlib is imported under that crate name from benches.
use _trimat::pack;
use _trimat::quantize;
use _trimat::tensor::TernaryTensor;
use _trimat::dispatch::best_kernel;

fn make_tensor(rows: usize, cols: usize) -> TernaryTensor {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 % 3.0) - 1.0)
        .collect();
    let (q, scale) = quantize::absmax_quantize(&data);
    let (nz, sg) = pack::encode(&q);
    TernaryTensor::new(rows, cols, nz, sg, vec![scale])
}

fn bench_gemm(c: &mut Criterion) {
    // (M, K, N): weight is M×K, x is K×N, output is M×N.
    let sizes = [
        (128usize, 256usize, 32usize),
        (512, 1024, 64),
        (1024, 4096, 128),
    ];
    let mut group = c.benchmark_group("gemm");

    for (m, k, n) in sizes {
        let w = make_tensor(m, k);
        let x: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.01).collect();
        let mut y = vec![0.0f32; m * n];
        let kernel = best_kernel();

        group.bench_with_input(
            BenchmarkId::new("trimat", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |b, _| {
                b.iter(|| kernel.gemm(&w, &x, n, &mut y));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_gemm);
criterion_main!(benches);
