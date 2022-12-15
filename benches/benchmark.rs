#![allow(unused)]
use std::fmt;
mod perf;

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration,
};
use matrix::vector::{DotProduct, LinearCombination, Vector, LinearInterpolation};

pub fn vector_add_benchmark(c: &mut Criterion) {
    let v1 = Vector::from_iter((0..10).map(|x| x as f32));
    let v2 = Vector::from_iter((0..10).map(|x| x as f32));

    c.bench_function("vector addition", |b| b.iter(|| &v1 + &v2));
}

pub fn linear_combination_f32_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_combination_f32");
    for n_vectors in (1..20).step_by(3) {
        for vector_size in (1..=1000).step_by(99) {
            let vectors: Vec<Vector<f32>> = (0..n_vectors)
                .map(|_| {
                    Vector::from(
                        (0..vector_size)
                            .map(|_| rand::random())
                            .collect::<Vec<f32>>(),
                    )
                })
                .collect();
            let coefficients: Vec<f32> = (0..n_vectors).map(|_| rand::random()).collect();
            group.bench_with_input(
                BenchmarkId::new(format!("linear_combination_f32_{n_vectors}"), vector_size),
                &vectors,
                |b, vectors| b.iter(|| vectors.linear_combination(&coefficients)),
            );
        }
    }
    group.finish();
}

pub fn linear_combination_f32_short_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_combination_f32_short");
    for n_vectors in [4] {
        for vector_size in [2, 4, 8, 10, 16, 20, 33, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] {
            let vectors: Vec<Vector<f32>> = (0..n_vectors)
                .map(|_| {
                    Vector::from(
                        (0..vector_size)
                            .map(|_| rand::random())
                            .collect::<Vec<f32>>(),
                    )
                })
                .collect();
            let coefficients: Vec<f32> = (0..n_vectors).map(|_| rand::random()).collect();
            group.bench_with_input(
                BenchmarkId::new(format!("n_vectors{n_vectors}"), vector_size),
                &vectors,
                |b, vectors| {
                    b.iter(|| black_box(vectors).linear_combination(black_box(&coefficients)))
                },
            );
        }
    }
    group.finish();
}

pub fn linear_combination_i32_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_combination_i32");
    for n_vectors in (1..100).step_by(10) {
        for vector_size in (1..=1000).step_by(99) {
            let vectors: Vec<Vector<i32>> = (0..n_vectors)
                .map(|_| {
                    Vector::from(
                        (0..vector_size)
                            .map(|_| rand::random())
                            .collect::<Vec<i32>>(),
                    )
                })
                .collect();
            let coefficients: Vec<i32> = (0..n_vectors).map(|_| rand::random()).collect();
            group.bench_with_input(
                BenchmarkId::new(format!("linear_combination_i32_{n_vectors}"), vector_size),
                &vectors,
                |b, vectors| b.iter(|| vectors.linear_combination(&coefficients)),
            );
        }
    }
    group.finish();
}

pub fn dot_product_i32_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product_i32");
    for i in (1..=1000).step_by(99) {
        let v1: Vector<i32> =
            Vector::from(&((0..=i).map(|_| rand::random::<i32>()).collect::<Vec<i32>>()));
        let v2 = Vector::from(&((0..=i).map(|_| rand::random::<i32>()).collect::<Vec<i32>>()));
        group.bench_with_input(
            BenchmarkId::new("dot_product_i32", i),
            &(v1, v2),
            |b, (v1, v2)| b.iter(|| v1.dot_product(v2)),
        );
    }
    group.finish();
}

pub fn dot_product_f32_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product_f32");
    for i in [1, 2, 4, 6, 8, 10, 16, 25, 50, 75, 100, 250, 500, 750, 1000] {
        let v1: Vector<f32> =
            Vector::from(&((0..=i).map(|_| rand::random::<f32>()).collect::<Vec<f32>>()));
        let v2 = Vector::from(&((0..=i).map(|_| rand::random::<f32>()).collect::<Vec<f32>>()));
        group.bench_with_input(
            BenchmarkId::new("dot_product_f32", i),
            &(v1, v2),
            |b, (v1, v2)| b.iter(|| v1.dot_product(v2)),
        );
    }
    group.finish();
}

pub fn linear_interpolation_f32_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("lerp_f32");
    for i in [1, 2, 4, 6, 8, 10, 16, 25, 50, 75, 100, 250, 300, 400, 500, 600, 750, 800, 900, 1000, 1500, 2000] {
        let v1: Vector<f32> =
            Vector::from(&((0..=i).map(|_| rand::random::<f32>()).collect::<Vec<f32>>()));
        let v2 = Vector::from(&((0..=i).map(|_| rand::random::<f32>()).collect::<Vec<f32>>()));
        group.bench_with_input(
            BenchmarkId::new("lerp_f32", i),
            &(v1, v2),
            |b, (v1, v2)| b.iter(|| v1.lerp(&v2, 0.5)),
        );
    }
    group.finish();
}


criterion_group!(
    name = benches;
    config = Criterion::default().with_plots().with_profiler(perf::FlamegraphProfiler::new(100));
    targets = //vector_add_benchmark,
    // linear_combination_f32_benchmark,
    // dot_product_f32_benchmark,
    // linear_combination_f32_short_benchmark,
    // linear_combination_i32_benchmark,
    // dot_product_i32_benchmark,
    linear_interpolation_f32_benchmark
);
criterion_main!(benches);
