pub mod dataset;
pub mod functions;
pub mod layers;
use dataset::{get_batch, load_images};
use functions::activations::{relu, softmax};
use layers::layer::{create_linear_layer, linear};
use mnist::*;
use ndarray::prelude::*;

fn main() {
    let (train_data, train_labels, test_data, test_labels) = load_images();
    let (w1, b1) = create_linear_layer(784, 512);

    let (w2, b2) = create_linear_layer(512, 10);

    let batch = get_batch(10, 0, train_data);

    let first_pass = linear(batch, w1, b1);
    let second_pass = linear(first_pass, w2, b2);
    let output = softmax(second_pass);
    let relued = relu(output);
    println!("Data output: {:?}", relued);

    let x = array![[1.0, 10.0], [3.0, 4.0]];
    let w = array![[2.0, 3.0, 3.9], [3.0, 5.0, 1.2],];
    let b = array![1.0, 1.0, 1.0];
    let value = linear(x, w, b);

    println!("{:?}", value);
}
