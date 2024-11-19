use ndarray::{
    arr1, arr2, arr3, array, linalg::general_mat_mul, Array, Array1, Array2, Array3, ArrayD, Axis,
    Dim,
};

use crate::{
    dataset::{argmax, get_batch, get_batch_1d},
    functions::{
        activations::{relu, softmax},
        loss::cross_entropy_loss,
    },
    layers::layer::{create_linear_layer, linear},
};

// Softmax gradient lil weird since it is higher dimensional
pub fn linear_backward(l: &mut Layer, gradient: &mut Array2<f32>, learning_rate: f32) {
    // weight derivative
    if l.input.is_none() {
        panic!("Layer needs to be forward passed before backwards pass can start");
    }
    let transposed_input = l.input.as_ref().unwrap().t().to_owned();
    // fel ordning igen?????
    let d_w = transposed_input.dot(gradient) / (l.input.as_ref().unwrap().dim().0 as f32);
    //println!("Weights dims: {:?}", l.weights.dim());
    //println!("inputs: {:?}", transposed_input);
    //println!("gradient: {:?}", gradient);

    // input derivate
    // Här med ???????
    let d_x = gradient.dot(&l.weights.t().to_owned());
    // Update weights and bias
    l.weights -= &(d_w * learning_rate);
    // Bias becomes a bit weird with dimensions
    //l.bias -= &(gradient.sum_axis(Axis(0)) * learning_rate);
    *gradient = d_x;
}
pub fn relu_backward(l: &mut Layer, gradient: &mut Array2<f32>) {
    let d_x = l
        .input
        .as_ref()
        .unwrap()
        .mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
    *gradient *= &d_x;
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
fn forward_pass(layers: &mut Vec<Layer>, data: &mut Array2<f32>, in_training: bool) {
    for layer in layers.iter_mut() {
        match layer.layer_type {
            LayerType::LINEAR => {
                // CLoning is very bad i know
                if in_training {
                    layer.input = Some(data.clone());
                }
                *data = linear(data.clone(), layer.weights.clone(), layer.bias.clone());
                if in_training {
                    layer.output = Some(data.clone());
                }
            }
            LayerType::RELU => {
                if in_training {
                    layer.input = Some(data.clone());
                }
                *data = relu(data.clone());
                if in_training {
                    layer.output = Some(data.clone());
                }
            }
            LayerType::SOFTMAX => {
                if in_training {
                    layer.input = Some(data.clone());
                }
                *data = softmax(data.clone());
                if in_training {
                    layer.output = Some(data.clone());
                }
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
    test_data: Array2<f32>,
    test_labels: Array1<f32>,
) {
    let mut gradient: Array2<f32> = Array2::zeros((batch_size as usize, 10));
    let (l1, relu, l2, softmax) = initialize_mlp();
    let mut layers = vec![l1, relu, l2, softmax];
    let batch_num = train_data.dim().0 as i32 / batch_size;
    for e in 0..epochs {
        println!("Epoch: {}", e);
        for batch in 0..batch_num {
            let mut batched_training_data = get_batch(batch_size, batch, &train_data);
            let batched_train_labels = get_batch(batch_size, batch, &train_labels);

            // forward pass
            forward_pass(&mut layers, &mut batched_training_data, true);
            let loss = cross_entropy_loss(&batched_training_data, &batched_train_labels);
            //println!("Training data output: {:?}", batched_training_data);
            println!("Loss: {:?}", loss.unwrap().sum() / batch_size as f32);
            // backward pass
            backward(
                &mut layers,
                learning_rate,
                &mut gradient,
                &batched_train_labels,
            );
        }
    }
    evaluate(&mut layers, test_data, test_labels, batch_size);
}

pub fn evaluate(
    layers: &mut Vec<Layer>,
    test_data: Array2<f32>,
    test_labels: Array1<f32>,
    batch_size: i32,
) {
    let mut correct_predictions = 0;

    let batch_num = test_data.dim().0 as i32 / batch_size;
    println!("Evaluation started: {:?}", test_data.dim());

    for batch in 0..batch_num {
        let mut batched_data = get_batch(batch_size, batch, &test_data);
        let batched_labels = get_batch_1d(batch_size, batch, &test_labels);

        forward_pass(layers, &mut batched_data, false);

        for (index, row) in batched_data.rows().into_iter().enumerate() {
            let prediction = argmax(&row.to_owned()).unwrap();

            if prediction == batched_labels[index] as usize {
                correct_predictions += 1;
            }
        }
    }

    println!("Correct predictions: {}", correct_predictions);
}
