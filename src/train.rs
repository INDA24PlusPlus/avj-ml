use ndarray::{
    arr1, arr2, arr3, array, linalg::general_mat_mul, Array, Array1, Array2, Array3, ArrayD, Dim,
};

use crate::{
    dataset::get_batch,
    functions::{activations::relu, loss::cross_entropy_loss},
    layers::layer::{create_linear_layer, linear},
};

// Softmax gradient lil weird since it is higher dimensional
pub fn linear_backward(l: &mut Layer, gradient: &mut Array2<f32>, learning_rate: f32) {
    // weight derivative
    if l.input.is_none() {
        panic!("Layer needs to be forward passed before backwards pass can start");
    }

    let transposed_input = l.input.as_ref().unwrap().t().to_owned();
    let d_w = transposed_input.dot(gradient);

    // input derivate
    let d_x = gradient.dot(&l.weights.t().to_owned());
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
// Fuse together cross entropy loss and softmax for calculating backwards, much easier that way
pub fn ce_softmax_backward(
    softmax_output: &Array2<f32>,
    labels: &Array2<f32>,
    gradient: &mut Array2<f32>,
) {
    *gradient = softmax_output - labels;
}

enum LayerType {
    LINEAR,
    RELU,
    SOFTMAX,
}
struct Layer {
    layer_type: LayerType,
    input: Option<Array2<f32>>,
    output: Option<Array2<f32>>,
    // These two will be empty if not linear layer
    weights: Array2<f32>,
    bias: Array1<f32>,
}

// TODO: optimize memory
fn forward_pass(layers: &mut Vec<Layer>, data: &mut Array2<f32>) {
    for layer in layers.iter_mut() {
        match layer.layer_type {
            LayerType::LINEAR => {
                // CLoning is very bad i know
                layer.input = Some(data.clone());
                *data = linear(data.clone(), layer.weights.clone(), layer.bias.clone());
                layer.output = Some(data.clone());
            }
            LayerType::RELU => {
                layer.input = Some(data.clone());
                *data = relu(data.clone());
                layer.output = Some(data.clone());
            }
            LayerType::SOFTMAX => {
                layer.input = Some(data.clone());
                *data = relu(data.clone());
                layer.output = Some(data.clone());
            }
        }
    }
}

fn backward(
    layers: &mut Vec<Layer>,
    learning_rate: f32,
    gradient: &mut Array2<f32>,
    labels: &Array2<f32>,
) {
    for layer in layers.iter_mut().rev() {
        match layer.layer_type {
            LayerType::LINEAR => {
                linear_backward(layer, gradient, learning_rate);
            }
            LayerType::RELU => {
                relu_backward(layer, gradient);
            }
            LayerType::SOFTMAX => {
                ce_softmax_backward(&layer.output.clone().unwrap(), &labels, gradient);
            }
        }
    }
}

fn initialize_mlp() -> (Layer, Layer, Layer, Layer) {
    let (w1, b1) = create_linear_layer(784, 512);
    let linear_layer_1: Layer = Layer {
        layer_type: LayerType::LINEAR,
        input: None,
        output: None,
        weights: w1,
        bias: b1,
    };

    let relu: Layer = Layer {
        layer_type: LayerType::RELU,
        input: None,
        output: None,
        weights: arr2(&[[]]),
        bias: arr1(&[]),
    };
    let (w2, b2) = create_linear_layer(512, 10);
    let linear_layer_2: Layer = Layer {
        layer_type: LayerType::LINEAR,
        input: None,
        output: None,
        weights: w2,
        bias: b2,
    };

    let softmax_layer: Layer = Layer {
        layer_type: LayerType::SOFTMAX,
        input: None,
        output: None,
        weights: arr2(&[[]]),
        bias: arr1(&[]),
    };

    return (linear_layer_1, relu, linear_layer_2, softmax_layer);
}

pub fn train_mlp(
    batch_size: i32,
    epochs: i32,
    learning_rate: f32,
    train_data: Array2<f32>,
    train_labels: Array2<f32>,
) {
    let mut gradient: Array2<f32> = Array2::zeros((batch_size as usize, 10));
    let (l1, relu, l2, softmax) = initialize_mlp();
    let mut layers = vec![l1, relu, l2, softmax];
    for e in 0..epochs {
        println!("Epoch: {}", e);
        for batch in 0..batch_size {
            let mut batched_training_data = get_batch(batch_size, batch, &train_data);
            let batched_train_labels = get_batch(batch_size, batch, &train_labels);

            // forward pass
            forward_pass(&mut layers, &mut batched_training_data);
            let loss = cross_entropy_loss(&batched_training_data, &batched_train_labels);
            println!("Training data output: {:?}", batched_training_data);
            println!("Loss: {:?}", loss.unwrap());
            // backward pass
            backward(
                &mut layers,
                learning_rate,
                &mut gradient,
                &batched_train_labels,
            );
        }
    }
}
