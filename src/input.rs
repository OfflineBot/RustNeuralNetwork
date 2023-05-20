use std::fs;
use std::io;

#[allow(unused)]
use ndarray::{array, Array1, Array2, Axis};

use crate::output;

pub struct Input {
    pub input: Array2<f64>,
    pub output: Array2<f64>,
}

#[allow(unused)]
// read mnist_images
pub fn read() -> Result<Input, io::Error> {
    if !check_exist() {
        return Err(io::Error::new(io::ErrorKind::NotFound, "Folder not found!"));
    }

    let folder = "mnist_images";
    let image_iteration_length = 10;
    let image_width = 28 * 28;

    let mut input_arr: Array2<f64> = Array2::zeros((image_iteration_length * 10, image_width));
    let mut output_arr: Array2<f64> = Array2::zeros((image_iteration_length * 10, 10));

    for i in 0..=9 {
        for a in 0..image_iteration_length {
            if let Ok(image) = image::open(format!("{}/{}/{}.png", folder, i, a)) {
                let gray_image = image.to_luma8();
                let image_array: Array1<f64> =
                    gray_image.pixels().map(|pixel| pixel[0] as f64).collect();

                input_arr
                    .index_axis_mut(Axis(0), a + i * 10)
                    .assign(&image_array);

                let mut out_arr1 = Array1::zeros(10);
                out_arr1[i] = 1 as f64;
                println!("outarr: {}\ni: {}", out_arr1, i);

                output_arr.index_axis_mut(Axis(0), a + i * 10).assign(&out_arr1);
                println!("OUTPUT_ARR: {}", output_arr);
            }
        }
    }
    
    output::print(input_arr.clone(), "input_arr.txt".to_owned());
    output::print(output_arr.clone(), "output_arr.txt".to_owned());
    Ok(Input {
        input: input_arr,
        output: output_arr,
    })
}

#[allow(unused)]
// get testing data
pub fn get_test_data() -> Result<Array2<f64>, io::Error> {
    //let output1: Array2<f64>;
    //for i in 0..=iteration {
        let folder = format!("input_data/0.png");
        if let Ok(image) = image::open(folder) {
            let gray_image = image.to_luma8();
            let image_array: Array1<f64> =
                gray_image.pixels().map(|pixel| pixel[0] as f64).collect();
            let mut out: Array2<f64> = Array2::zeros((1, 28 * 28));
            out.index_axis_mut(Axis(0), 0)
                .assign(&image_array.reversed_axes());
            return Ok(out);
        }
        Err(io::Error::new(io::ErrorKind::NotFound, "Data not Found!"))
    //}
}

fn check_exist() -> bool {
    if fs::metadata("mnist_images").is_ok() {
        return true;
    }
    false
}
