#![allow(unused)]
mod network;
mod input;
mod output;

use ndarray::{Array2, array};
use network::{Dimensions, Norm, Matrix, TrainData, Test};

fn main() {

    let x = input::read().unwrap();
    let y = input::get_test_data().unwrap();

    let test_input: Array2<f64> = array![[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]];
    let test_output: Array2<f64> = array![[1.0], [0.0], [1.0], [0.5]];
    let testing: Array2<f64> = array![[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.5]];
    let iterations = 10_000;
    let learning_rate = 0.01;


    let norm_data = Norm::normalize(x.input, x.output);
    
    let layers = Dimensions::new(784, 10, 10);
    let matrix = Matrix::new(layers);

    let training_matrix = TrainData::train(iterations, learning_rate, matrix.clone(), norm_data.clone());
    let output = Test::test(y, training_matrix.clone(), norm_data.clone());
    println!("OUT\n{}", output);
    output::print(output);
    output::print_training_data(training_matrix, norm_data);
}
