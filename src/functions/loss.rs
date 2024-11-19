use ndarray::{Array1, Array2};

// Predicted and reference: (B, D)
// Output will be (B)
pub fn cross_entropy_loss(
    predicted: &Array2<f32>,
    reference: &Array2<f32>,
) -> Result<Array2<f32>, &'static str> {
    if predicted.dim() != reference.dim() {
        Err("Dims must match, got")
    } else {
        let (batch_size, dimension) = predicted.dim();
        let mut batch_loss = Array2::from_shape_fn((batch_size, 1), |(i, j)| 0.0);
        for i in 0..batch_size {
            let log_predicted = predicted.row(i).map(|x| x.ln());
            let cross_entropy = Array1::from_elem(1, reference.dot(&log_predicted).sum());
            println!("CE shape: {:?}", cross_entropy);
            batch_loss.row_mut(i).assign(&cross_entropy);
        }
        return Ok(batch_loss);
    }
}
