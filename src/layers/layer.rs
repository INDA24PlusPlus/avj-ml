use std::f32::consts::PI;

use ndarray::{Array1, Array2};
use rand::prelude::*;

// Gaussian distribution with Box-Muller transform
fn generate_gaussian(mean: f32, standard_deviation: f32) -> f32 {
    let mut rng = thread_rng();
    let u1: f32 = rng.gen();
    let u2: f32 = rng.gen();
    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();

    return (z0 * standard_deviation) + mean;
}

pub fn create_linear_layer(in_size: usize, out_size: usize) -> (Array2<f32>, Array1<f32>) {
    let weights: Array2<f32> = Array2::from_shape_fn((in_size, out_size), |(i, j)| {
        generate_gaussian(0.0, (2.0 / (in_size as f32)).sqrt())
    });
    let bias: Array1<f32> = Array1::from_shape_fn(out_size, |i: usize| 0.0);

    return (weights, bias);
}
