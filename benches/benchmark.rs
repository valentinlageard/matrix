#![allow(unused)]
use std::fmt;
// mod perf;

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration,
};
use matrix::vector::{LinearCombination, Vector};

pub fn vector_add_benchmark(c: &mut Criterion) {
    let v1 = Vector::from(&(0..10).map(|x| x as f32).collect::<Vec<f32>>());
    let v2 = Vector::from(&(0..10).map(|x| x as f32).collect::<Vec<f32>>());

    c.bench_function("vector addition", |b| b.iter(|| &v1 + &v2));
}

pub fn linear_combination_f32_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_combination_f32");
    for n_vectors in (1..100).step_by(10) {
        for vector_size in (1..=1000).step_by(99) {
            let vectors: Vec<Vector<f32>> = (0..n_vectors)
                .map(|_| {
                    Vector::from(
                        (0..vector_size)
                            .map(|_| rand::random::<f32>())
                            .collect::<Vec<f32>>(),
                    )
                })
                .collect();
            let coefficients: Vec<f32> = (0..n_vectors).map(|_| rand::random::<f32>()).collect();
            group.bench_with_input(
                BenchmarkId::new(format!("linear_combination_f32_{n_vectors}"), vector_size),
                &vectors,
                |b, vectors| b.iter(|| vectors.linear_combination(&coefficients)),
            );
        }
    }
    group.finish();
}

pub fn linear_combination_i32_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_combination_i32");
    for i in (1..=1000).step_by(99) {
        let v1: Vector<i32> =
            Vector::from(&((0..=i).map(|_| rand::random::<i32>()).collect::<Vec<i32>>()));
        let v2 = Vector::from(&((0..=i).map(|_| rand::random::<i32>()).collect::<Vec<i32>>()));
        let vectors = [v1.clone(), v2.clone(), v1, v2];
        group.bench_with_input(
            BenchmarkId::new("linear_combination_i32", i),
            &vectors,
            |b, vectors| b.iter(|| vectors.linear_combination(&[10, -2, 10, 2])),
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().with_plots(); // .with_profiler(perf::FlamegraphProfiler::new(100))
    targets = vector_add_benchmark,
    linear_combination_f32_benchmark,
    linear_combination_i32_benchmark
);
criterion_main!(benches);
