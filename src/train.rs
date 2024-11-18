use ndarray::{arr3, array, Array1, Array2, Array3};

use crate::dataset::get_batch;

pub fn linear_backward(l: &mut Layer, gradient: &mut Array2<f32>, learning_rate: f32) {
    // weight derivative
    let transposed_input = l.input.t();
    let d_w = gradient.dot(&transposed_input);

    // input derivate
    let d_x = l.weights.t().dot(gradient);

    // Update weights and bias
    l.weights -= &(d_w * learning_rate);
    // Bias becomes a bit weird with dimensions
    //l.bias -= &(&*gradient * learning_rate);
    *gradient = d_x;
}
pub fn relu_backward(l: &mut Layer, gradient: &mut Array2<f32>) {
    let d_x = gradient.map(|x| if *x > 0.0 { 1.0 } else { 0.0 });
    *gradient = d_x;
}
pub fn cross_entropy_backwards(
    inputs: Array2<f32>,
    gradient: &mut Array2<f32>,
    train_labels: Array2<f32>,
) {
    let ce_gradient =
        Array2::from_shape_fn(inputs.dim(), |(i, j)| train_labels[[i, j]] / inputs[[i, j]]);

    *gradient = ce_gradient;
}
pub fn softmax_backward(l: &mut Layer, gradient: &mut Array3<f32>) {
    let softmax_derivative = Array3::from_shape_fn(
        (l.input.dim().0, l.input.dim().1, l.input.dim().1),
        |(i, j, k)| {
            if j == k {
                l.output[[i, j]] * (1.0 - l.output[[i, j]])
            } else {
                -l.output[[i, j]] * l.output[[i, k]]
            }
        },
    );
    *gradient = softmax_derivative;
}

enum LayerType {
    LINEAR,
    RELU,
    SOFTMAX,
}
struct Layer {
    layer_type: LayerType,
    input: Array2<f32>,
    output: Array2<f32>,
    // These two will be empty if not linear layer
    weights: Array2<f32>,
    bias: Array2<f32>,
}

pub fn train_mlp(
    batch_size: i32,
    epochs: i32,
    learning_rate: f32,
    train_data: Array2<f32>,
    train_labels: Array2<f32>,
) {
    let gradient: Array2<f32> = Array2::zeros((batch_size as usize, 10));
    // very hacky but softmax gradient will be 3d because of batching and we need to take care of that
    let softmax_gradient = arr3(&[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]]);

    for e in 0..epochs {
        for batch in 0..batch_size {
            let batched_training_data = get_batch(batch_size, batch, &train_data);
            let batched_train_labels = get_batch(batch_size, batch, &train_labels);

            // forward pass

            // backward pass
        }
    }
}
