use mnist::{Mnist, MnistBuilder};
use ndarray::prelude::*;

pub fn load_images() -> (Array2<f32>, Array2<f32>, Array2<f32>, Array1<f32>) {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let raw_train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);
    let train_data = format_images(raw_train_data);

    // Convert the returned Mnist struct to Array2 format
    let raw_train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);
    let train_labels = one_hot_labels(raw_train_labels);

    let raw_test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);
    let test_data = format_images(raw_test_data);

    let test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    (
        train_data,
        train_labels,
        test_data,
        test_labels.flatten().to_owned(),
    )
}

// Assume data: (N, W, H)
// Convert into (N, W*H)
pub fn format_images(data: Array3<f32>) -> Array2<f32> {
    let N = data.shape()[0];
    let W = data.shape()[1];
    let H = data.shape()[2];
    let flattened_images = data
        .into_shape_with_order((N, W * H))
        .expect("Could not reshape");
    // flattened_images: (N, W*H)
    return flattened_images;
}

// Assume labels: (N, 1)
// reshape into: (N, 10)
pub fn one_hot_labels(labels: Array2<f32>) -> Array2<f32> {
    let (n, _) = labels.dim();
    let mut output = Array2::zeros((n, 10));

    for (i, &value) in labels.iter().enumerate() {
        if value < 10.0 {
            output[[i, value as usize]] = 1.0;
        }
    }

    output
}

// Assume data is of dims: (N, D)
// Return: (B, T) where B is batch_size
pub fn get_batch(batch_size: i32, batch_num: i32, data: &Array2<f32>) -> Array2<f32> {
    let start = batch_size * batch_num;
    let end = batch_size * (batch_num + 1);

    return data.slice(s![start..end, ..]).to_owned();
}

pub fn get_batch_1d(batch_size: i32, batch_num: i32, data: &Array1<f32>) -> Array1<f32> {
    let start = batch_size * batch_num;
    let end = batch_size * (batch_num + 1);

    return data.slice(s![start..end]).to_owned();
}

// Find the index of the activation that is the largest

pub fn argmax(array: &Array1<f32>) -> Option<usize> {
    if array.is_empty() {
        return None; // Return None for empty arrays
    }
    let mut max_index = 0;
    let mut max_value = &array[0];

    for (i, value) in array.iter().enumerate() {
        if value > max_value {
            max_value = value;
            max_index = i;
        }
    }

    Some(max_index)
}
