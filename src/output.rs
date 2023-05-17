use std::fs::File;
use std::io::prelude::*;
use ndarray::Array2;

use crate::network::{Matrix, Norm};

pub fn print(data: Array2<f64>) {
    let mut file = File::create("my_data.txt").unwrap();
    file.write_all(format!("{}", data).as_bytes()).unwrap();
}

pub fn print_training_data(data: Matrix, norm: Norm) {
    let mut file = File::create("training_data.txt").unwrap();
    file.write_all(format!("W1:{};", data.w1).as_bytes());
    file.write_all(
        format!("W2:{};\nB1:{};\nB2:{};\nXMEAN:{};\nYMEAN:{};\nXSTD:{};\nYSTD:{};\n", 
        data.w2, data.b1, data.b2, norm.input_mean, norm.output_mean, norm.input_std, norm.output_std)
        .as_bytes()
    );
}