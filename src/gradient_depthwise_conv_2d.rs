use crate::{
    activation::FusedActivation,
    buffer::{Buffer2D, Buffer4D},
    quantize::{quantize, Trainable},
    tensor::{Tensor4D, TensorView, TensorViewPadding},
    update_layer::{accumulate_gradient_4D, clip_norm_4D, get_input_index},
};
use core::array;
use nalgebra::{SMatrix, SVector};

pub fn update_grad_depthwise_conv_2d<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const WEIGHTS_CHANS: usize,
    const FILTER_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS, FILTER_QUANTS>,
    weights_gradient: &mut Buffer4D<i32, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS>,
    constants: &(
        Buffer2D<f32, WEIGHTS_CHANS, 1>,
        Buffer2D<f32, FILTER_QUANTS, 1>,
    ),
    constants_gradient: &mut (
        Buffer2D<f32, WEIGHTS_CHANS, 1>,
        Buffer2D<f32, FILTER_QUANTS, 1>,
    ),
    outputs: Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS, 1>,
    output_grad: Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS>,
    activation: FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
    bias_scale: [f32; FILTER_QUANTS],
    learning_rate: f32,
    backwards_clip_val: f32,
) -> Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> {
    let output_grad = clip_norm_4D(&output_grad, backwards_clip_val);
    let grad_weight = grad_depthwise_conv_2d_weights(
        input,
        weights,
        &outputs,
        &output_grad,
        &activation,
        strides,
        padding,
    );
    accumulate_gradient_4D(&grad_weight, weights_gradient);
    let grad_bias = grad_depthwise_conv_2d_bias(
        input,
        weights,
        &outputs,
        &output_grad,
        &activation,
        bias_scale,
    );
    update_bias_dephtwise_conv_2d(constants_gradient, grad_bias);
    grad_depthwise_conv_2d_inputs(
        input,
        weights,
        &outputs,
        &output_grad,
        &activation,
        strides,
        padding,
    )
}
pub fn update_bias_dephtwise_conv_2d<const FILTER_QUANTS: usize, const WEIGHT_CHANS: usize>(
    constants: &mut (
        Buffer2D<f32, WEIGHT_CHANS, 1>,
        Buffer2D<f32, FILTER_QUANTS, 1>,
    ),
    bias_gradient: Buffer2D<f32, WEIGHT_CHANS, 1>,
) {
    constants.0 = SMatrix::from_fn(|i, j| constants.0[(i, j)] + bias_gradient[(i, j)]);
}
pub fn grad_depthwise_conv_2d_inputs<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const WEIGHTS_CHANS: usize,
    const FILTER_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS, FILTER_QUANTS>,
    outputs: &Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS, 1>,
    output_grad: &Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS>,
    activation: &FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
) -> Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> {
    let mut accum: Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| [0i32; INPUT_CHANS]));
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    let normalization_param = output_grad.iter().fold(0f32, |acc, val| {
        acc + val.iter().fold(0f32, |acc1, val1| {
            acc1 + val1
                .iter()
                .fold(0f32, |acc2, val2| acc2 + val2.abs() as f32)
        })
    }) as f32;
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
            let coord = get_input_index(
                WEIGHTS_ROWS,
                WEIGHTS_COLS,
                (output_row, output_col),
                padding,
                strides,
            );
            for output_channel in 0..WEIGHTS_CHANS {
                let val = outputs.buffer[0][(output_row, output_col)][output_channel]
                    .saturating_sub(outputs.zero_point[0]);
                if !(match activation {
                    FusedActivation::Relu => val > T::zero(),
                    FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                    _ => true,
                }) {
                    continue;
                }
                for filter_row in 0..WEIGHTS_ROWS {
                    if (coord.0 + filter_row as i32) < 0
                        || (coord.0 + filter_row as i32) >= INPUT_ROWS as i32
                    {
                        continue;
                    }
                    for filter_col in 0..WEIGHTS_COLS {
                        if (coord.1 + filter_col as i32) < 0
                            || (coord.1 + filter_col as i32) >= INPUT_COLS as i32
                        {
                            continue;
                        }
                        let cur_coord = (
                            (coord.0 + filter_row as i32) as usize,
                            (coord.1 + filter_col as i32) as usize,
                        );
                        let zero_point: i32 = weights
                            .zero_point
                            .get(output_channel)
                            .copied()
                            .unwrap_or(weights.zero_point[0])
                            .to_superset();
                        let tmp: i32 = weights.buffer[0][(filter_row, filter_col)][output_channel]
                            .to_superset();
                        accum[0][cur_coord][if output_channel < INPUT_CHANS {
                            output_channel
                        } else {
                            0
                        }] += (tmp - zero_point)
                            * output_grad[0][(output_row, output_col)][output_channel]
                    }
                }
            }
        }
    }
    accum.map(|batch| {
        batch.map(|channels| channels.map(|ch| (ch as f32 / normalization_param).round() as i32))
    })
}
pub fn grad_depthwise_conv_2d_weights<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const WEIGHTS_CHANS: usize,
    const WEIGHTS_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS, WEIGHTS_QUANTS>,
    outputs: &Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS, 1>,
    output_grad: &Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS>,
    activation: &FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
) -> Buffer4D<i32, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS> {
    let mut accum: Buffer4D<i32, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| [0i32; WEIGHTS_CHANS]));
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
            let view: TensorView<T, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> =
                input.view((output_row, output_col), 0, padding, strides);
            for output_channel in 0..WEIGHTS_CHANS {
                let val = outputs.buffer[0][(output_row, output_col)][output_channel]
                    .saturating_sub(outputs.zero_point[0]);
                if !(match activation {
                    FusedActivation::Relu => val > T::zero(),
                    FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                    _ => true,
                }) {
                    continue;
                }
                for filter_row in 0..WEIGHTS_ROWS {
                    for filter_cols in 0..WEIGHTS_COLS {
                        let zero_point: i32 = input.zero_point[0].to_superset();
                        if view.mask[(filter_row, filter_cols)] {
                            let tmp: i32 = view.buffer[(filter_row, filter_cols)][if output_channel
                                < INPUT_CHANS
                            {
                                output_channel
                            } else {
                                0
                            }]
                            .to_superset();
                            accum[0][(filter_row, filter_cols)][output_channel] += (tmp
                                - zero_point)
                                * &output_grad[0][(output_row, output_col)][output_channel];
                        }
                    }
                }
            }
        }
    }
    accum
}
pub fn grad_depthwise_conv_2d_bias<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const WEIGHTS_CHANS: usize,
    const FILTERS_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS, FILTERS_QUANTS>,
    outputs: &Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS, 1>,
    output_grad: &Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS>,
    activation: &FusedActivation,
    bias_scale: [f32; FILTERS_QUANTS],
) -> SVector<f32, WEIGHTS_CHANS> {
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    let mut accum: SVector<i32, WEIGHTS_CHANS> = SVector::zeros();
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
            for output_batch in 0..WEIGHTS_CHANS {
                let val = outputs.buffer[0][(output_row, output_col)][output_batch]
                    .saturating_sub(outputs.zero_point[0]);
                if !(match activation {
                    FusedActivation::Relu => val > T::zero(),
                    FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                    _ => true,
                }) {
                    continue;
                }
                accum[output_batch] += output_grad[0][(output_row, output_col)][output_batch];
            }
        }
    }
    SMatrix::from_fn(|i, _| {
        // let filters_scale = weights.scale.get(i).copied().unwrap_or(weights.scale[0]);
        // let bias_scale_cur = bias_scale.get(i).copied().unwrap_or(bias_scale[0]);
        // let scale = bias_scale_cur / (filters_scale * input.scale[0]).powi(2);
        let tmp: f32 = accum[i] as f32;
        tmp * bias_scale.get(i).unwrap_or(&bias_scale[0])
    })
}
