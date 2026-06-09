use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

// Access internals through the trimat crate name (defined in [package] name).
// The cdylib target is "_trimat" for Python, but the rlib is "trimat" for benches.
use trimat::pack;
use trimat::quantize;
use trimat::tensor::TernaryTensor;
use trimat::dispatch::best_kernel;

fn make_tensor(rows: usize, cols: usize) -> TernaryTensor {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 % 3.0) - 1.0)
        .collect();
    let (q, scale) = quantize::absmax_quantize(&data);
    let (nz, sg) = pack::encode(&q);
    TernaryTensor::new(rows, cols, nz, sg, vec![scale])
}

fn bench_gemv(c: &mut Criterion) {
    let sizes = [(128usize, 256usize), (512, 1024), (1024, 4096)];
    let mut group = c.benchmark_group("gemv");

    for (rows, cols) in sizes {
        let w = make_tensor(rows, cols);
        let x: Vec<f32> = (0..cols).map(|i| i as f32 * 0.01).collect();
        let mut y = vec![0.0f32; rows];
        let kernel = best_kernel();

        group.bench_with_input(
            BenchmarkId::new("trimat", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, _| {
                b.iter(|| kernel.gemv(&w, &x, &mut y));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_gemv);
criterion_main!(benches);
