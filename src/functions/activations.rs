use ndarray::{Array2, ArrayBase, Axis, Dim, OwnedRepr};

// B = Batch size
// T = Vector size
// data = B x T

pub fn softmax(data: Array2<f32>) -> Array2<f32> {
    let (B, _) = data.dim();
    let mut softmax_activations = data.clone();
    for i in 0..B {
        println!("Softmax pre-activations {:?}", softmax_activations);
        let row_sum = data.row(i).map(|x| x.exp()).sum();
        let row_activation = data.row(i).map(|x| x.exp() / row_sum);
        println!("Value of activations: {:?}", row_activation);
        softmax_activations.row_mut(i).assign(&row_activation);
    }

    return softmax_activations;
}

// B = Batch size
// T = Vector size
// data = B x T
pub fn relu(data: Array2<f32>) -> Array2<f32> {
    let relu_applied: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = data.map(|x| {
        if *x > 0.0 {
            return *x;
        } else {
            return 0.0;
        }
    });

    return relu_applied;
}

#[cfg(test)]
mod tests {
    // softmax tests
}
