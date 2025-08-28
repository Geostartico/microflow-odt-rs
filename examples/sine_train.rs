use libm::sinf;
use microflow_train_macros::model;
use nalgebra::{matrix, SMatrix};
use rand::Rng;
use std::{fs::File, io::Write};
#[model("models/train/sine.tflite", 2, "mse", false, [16000.0,0.0],[1024.0,256.0])]
struct Sine {}

fn main() {
    let mut model = Sine::new();
    let mut rng = rand::rng();
    let output_scale = 0.00785118155181408;
    let output_zero_point = 1i8;
    let epochs = 500;
    let samples = 1000;
    let batch = 16;
    let learning_rate = 0.01;
    let mut output_file = File::create("output_sine_untrained").unwrap();
    for sample in 0..samples {
        let x = rng.random_range(0.0..(2.0 * std::f32::consts::PI));
        let y = sinf(x);
        let output = model.predict(matrix![x]);
        let output_string = format!("{} {} {}\n", x, y, output[0]);
        output_file.write_all(output_string.as_bytes()).unwrap();
    }
    println!("finished printing baseline");
    let mut counter = 0;
    let initial = model.weights0.buffer.clone();
    println!("initial_weights: {}", initial);
    'epochs: for e in 0..epochs {
        println!("epoch {}", e);
        for sample in 0..samples {
            let x = rng.random_range(0.0..(2.0 * std::f32::consts::PI));
            // let y = 0.5 * sinf(x);
            let y = x / 4f32;
            // println!("x unquantized: {x}");
            // println!("y unquantized: {y}");
            let output = microflow::tensor::Tensor2D::quantize(
                matrix![y],
                [output_scale],
                [output_zero_point],
            );
            // println!("y quantized: {}", output.buffer[0]);

            let y_p = model.predict_train(matrix![x], &output, learning_rate)[0];
            // println!("predicted: {y_p}");
            // println!(
            // "predicted quantized: {}",
            // microflow::quantize::quantize(y_p, output_scale, output_zero_point)
            // );
            // println!("batch back gradient: {}", model.weights0_gradient);
            // println!("++++++++++++++++++++++++++++");
            if (sample + 1) % batch == 0 {
                // println!("batch back gradient: {}", model.weights0_gradient);

                // microflow::update_layer::update_weights_perc_2D::<i8, 16, 1, 4>(
                //     &mut model.weights0,
                //     &model.weights0_gradient,
                //     batch,
                //     learning_rate,
                // );
                // println!("final_gradient: {}", model.weights0_gradient);
                // println!("final_weights: {}", model.weights0.buffer);
                // println!(
                //     "model batch gradient = {}",
                //     // learning_rate * model.weights0_gradient.cast::<f32>() / batch as f32
                //     model.weights0_gradient
                // );
                // println!(
                //     "final_weights difference: {}",
                //     model.weights0.buffer.cast::<i32>() - initial.cast::<i32>()
                // );
                // if counter == 0 {
                //     break 'epochs;
                // }
                counter += 1;
                model.update_layers(batch as usize, learning_rate);
                // panic!();
            }
        }
    }
    println!("final_weights: {}", model.weights0.buffer);
    println!(
        "final_weights difference: {}",
        model.weights0.buffer.cast::<i32>() - initial.cast::<i32>()
    );
    let mut output_file = File::create("output_sine_trained").unwrap();
    for sample in 0..samples {
        let x = rng.random_range(0.0..(2.0 * std::f32::consts::PI));
        let y = 0.5 * sinf(x);
        let output = model.predict(matrix![x]);
        let output_string = format!("{} {} {}\n", x, y, output[0]);
        output_file.write_all(output_string.as_bytes()).unwrap();
    }
}
