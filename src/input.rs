use std::fs;
use std::io;

use ndarray::{Array2, Axis, Array1, array};

pub struct Input {
    pub input: Array2<f64>,
    pub output: Array2<f64>,
}

pub fn read() -> Result<Input, io::Error> {
    if !check_exist() {
        return Err(io::Error::new(io::ErrorKind::NotFound, "Folder not found!"));
    }

    let folder = "data";
    let image_iteration_length = 1;
    let image_width = 28 * 28;

    let mut input_arr: Array2<f64> = Array2::zeros((image_iteration_length * 10, image_width));
    let mut output_arr: Array2<f64> = Array2::zeros((image_iteration_length * 10, 10));

    for i in 0..=9 {
        for a in 0..=image_iteration_length {
            if let Ok(image) = image::open(format!("{}/{}/{}.png", folder, i, a)) {
                let gray_image = image.to_luma8();
                let image_array: Array1<f64> = gray_image.pixels().map(|pixel| pixel[0] as f64).collect();
                input_arr.index_axis_mut(Axis(0), a + i).assign(&image_array);

                let mut out_arr1 = Array1::zeros(10);
                out_arr1[i] = 1 as f64;
                output_arr.index_axis_mut(Axis(0), a + i).assign(&out_arr1);
            }
        }
    }

    Ok(Input { 
        input: input_arr, 
        output: output_arr, 
    })
}

pub fn get_test_data() -> Result<Array2<f64>, io::Error> {
    let folder = "input_data/0.png";
    if let Ok(image) = image::open(folder) {
        let gray_image = image.to_luma8();
        let image_array: Array1<f64> = gray_image.pixels().map(|pixel| pixel[0] as f64).collect();
        let out: Array2<f64> = Array2::zeros((1, 28 * 28));
        return Ok(out);
    } 
    Err(
        io::Error::new(io::ErrorKind::NotFound, "Data not Found!")
    )
}

fn check_exist() -> bool {
    if fs::metadata("data").is_ok() {
        return true;
    }
    false
}