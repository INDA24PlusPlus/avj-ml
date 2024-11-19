pub mod dataset;
pub mod functions;
pub mod layers;
pub mod train;
use dataset::{get_batch, load_images};
use functions::{
    activations::{relu, softmax},
    loss::cross_entropy_loss,
};
use layers::layer::{create_linear_layer, linear};
use ndarray::prelude::*;
use train::train_mlp;

fn main() {
    let (train_data, train_labels, test_data, test_labels) = load_images();

    train_mlp(11, 3, 0.001, train_data, train_labels);
}
