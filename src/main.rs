mod network;

use ndarray::{Array2, array};
use network::{Dimensions, Norm, Matrix, TrainData, Test};

fn main() {

    let test_input: Array2<f64> = array![[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]];
    let test_output: Array2<f64> = array![[1.0], [0.0], [1.0], [0.5]];
    let testing: Array2<f64> = array![[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.5]];
    let iterations = 100_000;
    let learning_rate = 0.01;


    let norm_data = Norm::normalize(test_input, test_output);
    
    let layers = Dimensions::new(2, 10, 1);
    let matrix = Matrix::new(layers);

    let training_matrix = TrainData::train(iterations, learning_rate, matrix.clone(), norm_data.clone());
    let output = Test::test(testing, training_matrix, norm_data);
    println!("OUT\n{}", output);
}
