use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

struct Activation;
impl Activation {
    fn sigmoid(x: Array2<f64>) -> Array2<f64> {
        1.0 / (1.0 + (-x).mapv(|y| y.exp()))
    }
    fn sigmoid_prime(x: Array2<f64>) -> Array2<f64> {
        let sig = Activation::sigmoid(x);
        sig.clone() * (1.0 - sig)
    }
}

#[derive(Clone)]
pub struct Norm {
    pub input_data: Array2<f64>,
    pub input_mean: Array1<f64>,
    pub input_std: Array1<f64>,
    pub input_norm: Array2<f64>,
    pub output_data: Array2<f64>,
    pub output_mean: Array1<f64>,
    pub output_std: Array1<f64>,
    pub output_norm: Array2<f64>,
}
struct MiniNorm {
    mean: Array1<f64>,
    std: Array1<f64>,
    norm: Array2<f64>,
}
impl Norm {
    pub fn normalize(input: Array2<f64>, output: Array2<f64>) -> Norm {
        let x = Norm::normalize_data(input.clone());
        let y = Norm::normalize_data(output.clone());
        Norm {
            input_data: input,
            input_mean: x.mean,
            input_std: x.std,
            input_norm: x.norm,
            output_data: output,
            output_mean: y.mean,
            output_std: y.std,
            output_norm: y.norm,
        }
    }
    fn normalize_data(data: Array2<f64>) -> MiniNorm {
        let mean = data.mean_axis(Axis(0)).unwrap();
        let pre_std = data.std_axis(Axis(0), 0.0);
        let std = pre_std.mapv(|mut x| {
            if x == 0.0 {
                x = 1e-10;
            }
            x
        });
        let norm = (data - mean.clone()) / std.clone();
        MiniNorm { mean, std, norm }
    }
}

pub struct Dimensions {
    input_layer_size: usize,
    hidden_layer_size: usize,
    output_layer_size: usize,
}
impl Dimensions {
    pub fn new(input_layer: usize, hidden_layer: usize, output_layer: usize) -> Self {
        Dimensions {
            input_layer_size: input_layer,
            hidden_layer_size: hidden_layer,
            output_layer_size: output_layer,
        }
    }
}

#[derive(Clone)]
pub struct Matrix {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
}
impl Matrix {
    pub fn new(dimension: Dimensions) -> Matrix {
        let w1 = Array2::random(
            (dimension.input_layer_size, dimension.hidden_layer_size),
            Uniform::new(-1.0, 1.0),
        );
        let b1 = Array1::random(dimension.hidden_layer_size, Uniform::new(-1.0, 1.0));
        let w2 = Array2::random(
            (dimension.hidden_layer_size, dimension.output_layer_size),
            Uniform::new(-1.0, 1.0),
        );
        let b2 = Array1::random(dimension.output_layer_size, Uniform::new(-1.0, 1.0));
        Matrix { w1, b1, w2, b2 }
    }
}

pub struct TrainData;
impl TrainData {
    pub fn train(iterations: usize, learning_rate: f64, mut matrix: Matrix, data: Norm) -> Matrix {

        for i in 0..=iterations {
            
            if i % 1_000 == 0 {
                println!("{}", (iterations - i) / 1_000 );
            }

            let z1 = data.input_norm.dot(&matrix.w1) + &matrix.b1;
            let a1 = Activation::sigmoid(z1.clone());
            let z2 = a1.dot(&matrix.w2) + &matrix.b2;
            let y_pred = z2;
    
            let delta2 = y_pred - data.output_norm.clone();
            let delta1 = delta2.dot(&matrix.w2.t()) * Activation::sigmoid_prime(z1);

            matrix.w1 = &matrix.w1 - data.input_norm.t().dot(&delta1) * learning_rate;
            matrix.b1 = &matrix.b1 - delta1.sum_axis(Axis(0)) * learning_rate;
            matrix.w2 = &matrix.w2 - a1.t().dot(&delta2) * learning_rate;
            matrix.b2 = &matrix.b2 - delta2.sum_axis(Axis(0)) * learning_rate;
        }

        matrix
    }
}

pub struct Test;
impl Test {
    pub fn test(test: Array2<f64>, matrix: Matrix, data: Norm) -> Array2<f64> {
        let test_input_norm = (test - data.input_mean) / data.input_std;
        let z1 = test_input_norm.dot(&matrix.w1) + &matrix.b1;
        let a1 = Activation::sigmoid(z1.clone());
        let z2 = a1.dot(&matrix.w2) + &matrix.b2;
        let test_output = z2 * data.output_std + data.output_mean;
        test_output
    }
}