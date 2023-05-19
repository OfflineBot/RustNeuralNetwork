
mod network;
mod input;
mod output;

use network::{Dimensions, Norm, Matrix, TrainData, Test};

fn main() {

    let x = input::read().unwrap();
    let y = input::get_test_data().unwrap();
    
    //let test_input: Array2<f64> = array![[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]];
    //let test_output: Array2<f64> = array![[1.0], [0.0], [1.0], [0.5]];
    //let testing: Array2<f64> = array![[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.5]];
    let iterations = 100;
    let learning_rate = 0.0001;

    let norm_data = Norm::normalize(x.input, x.output);
    //println!("norm_data: {:?}", norm_data);
    
    let layers = Dimensions::new(784, 100, 10);
    let matrix = Matrix::new(layers);

    let training_matrix = TrainData::train(iterations, learning_rate, matrix.clone(), norm_data.clone());
    //println!("---------\ntrainings_data: {:?}", training_matrix.b1);
    let output = Test::test(y, training_matrix.clone(), norm_data.clone());

    println!("OUT\n{}", output);
    output::print(output);
    //println!("Should be: \n{}", mnist.3);
    output::print_training_data(training_matrix, norm_data);
}
