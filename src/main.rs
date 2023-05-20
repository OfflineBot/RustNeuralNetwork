
mod network;
mod input;
mod output;

use network::{Dimensions, Norm, Matrix, TrainData, Test};

fn main() {
    let image_use = 1000;
    let x = input::read(image_use).unwrap();
    let y = input::get_test_data().unwrap();
    

    let iterations = 1_000;
    let learning_rate = 0.000001;
    let print_size = 10;

    println!("Running Network");
    let norm_data = Norm::normalize(x.input, x.output);
    
    let layers = Dimensions::new(784, 100, 10);
    let matrix = Matrix::new(layers);

    let training_matrix = TrainData::train(iterations, learning_rate, matrix.clone(), norm_data.clone(), print_size);
    let output = Test::test(y, training_matrix.clone(), norm_data.clone());

    println!("OUT\n{}", output);
    output::print(output, "my_data.txt".to_owned());

    output::print_training_data(training_matrix, norm_data);
}
