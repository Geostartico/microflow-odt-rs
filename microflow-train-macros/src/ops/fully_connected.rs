use crate::activation::TokenFusedActivation;
use crate::buffer::TokenBuffer2D;
use crate::quantize::{TokenQuantized, TrainToTokens};
use crate::tensor::TokenTensor2D;
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::{convert_ref, DMatrix};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use simba::scalar::SupersetOf;
use syn::{parse_quote, ItemStruct};

/// Represents the tokenized version of the `FullyConnected` operator.
pub(crate) struct TokenFullyConnected<T: TokenQuantized> {
    pub(crate) weights: TokenTensor2D<T>,
    pub(crate) output: TokenTensor2D<T>,
    pub(crate) input: Option<Vec<usize>>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) constants: (TokenBuffer2D<f32>, f32, TokenBuffer2D<i32>, i32),
    pub(crate) constants_gradient: (TokenBuffer2D<f32>, f32, TokenBuffer2D<i32>, i32),
    pub(crate) index: usize,
    pub(crate) reshape: bool,
    pub(crate) scale_bias: f32,
    pub(crate) layer_index: i32,
    pub(crate) input_zero_point: T,
    pub(crate) train: bool,
    pub(crate) back_norm: f32,
    pub(crate) gradient_norm: f32,
}

/// Parses the [`TokenFullyConnected`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
/// * `buffers` - The model buffers as a [`Vector<ForwardsUOffset<Buffer>>`]
/// * `index` - The operator index
/// * `layer_index` - index of the layer, used for the training pipeline
///
pub(crate) fn parse_indexed(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
    index: usize,
    layer_index: i32,
    back_norm: f32,
    gradient_norm: f32,
) -> Box<dyn TrainToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenFullyConnected::<i8>::new(
            operator,
            tensors,
            buffers,
            index,
            layer_index,
            back_norm,
            gradient_norm,
        )),
        TensorType::UINT8 => Box::new(TokenFullyConnected::<u8>::new(
            operator,
            tensors,
            buffers,
            index,
            layer_index,
            back_norm,
            gradient_norm,
        )),
        _ => unimplemented!(),
    }
}

/// Parses the [`TokenFullyConnected`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
/// * `buffers` - The model buffers as a [`Vector<ForwardsUOffset<Buffer>>`]
/// * `index` - The operator index
///
pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
    index: usize,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenFullyConnected::<i8>::new(
            operator, tensors, buffers, index, -1, -1f32, -1f32,
        )),
        TensorType::UINT8 => Box::new(TokenFullyConnected::<u8>::new(
            operator, tensors, buffers, index, -1, -1f32, -1f32,
        )),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenFullyConnected<T> {
    /// Builds the [`TokenFullyConnected`] operator from the given model operator and tensors.
    ///
    /// # Arguments
    /// * `operator` - The model operator as an [`Operator`]
    /// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
    /// * `buffers` - The model buffers as a [`Vector<ForwardsUOffset<Buffer>>`]
    /// * `index` - The operator index
    ///
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
        index: usize,
        layer_index: i32,
        back_norm: f32,
        gradient_norm: f32,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        let input = TokenTensor2D::from_empty_tensor(tensors.get(inputs.get(0) as usize));
        let weights =
            TokenTensor2D::from_buffered_tensor(tensors.get(inputs.get(1) as usize), buffers);
        let output = TokenTensor2D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let biases = if inputs.get(2) < 0 {
            let num_bias = output.shape.get(1).unwrap();
            TokenTensor2D::zeroed_tensor((*num_bias, 1))
        } else {
            TokenTensor2D::from_buffered_tensor(tensors.get(inputs.get(2) as usize), buffers)
        };
        let options = operator
            .builtin_options_as_fully_connected_options()
            .unwrap();
        let constants = Self::preprocess(&input, &weights, &biases, &output);
        let constants_gradient = Self::zero_constants(&weights, &biases);
        Self {
            weights,
            output,
            input: if input.shape.len() != 2 {
                Option::Some(input.shape.clone())
            } else {
                Option::None
            },
            constants_gradient,
            fused_activation: options.fused_activation_function().into(),
            reshape: input.shape.len() != 2,
            scale_bias: biases.scale[0],
            constants,
            index,
            layer_index,
            input_zero_point: *(input.zero_point.get(0).unwrap()),
            train: false,
            back_norm,
            gradient_norm,
        }
    }

    /// Pre-processes the operator, returning the tuple of constants.
    ///
    /// # Arguments
    /// * `input` - The input of the operator as a [`TokenTensor2D`]
    /// * `weights` - The weights of the operator as a [`TokenTensor2D`]
    /// * `biases` - The biases of the operator as a [`TokenTensor2D`]
    /// * `output` - The output of the operator as a [`TokenTensor2D`]
    ///
    fn preprocess(
        input: &TokenTensor2D<T>,
        weights: &TokenTensor2D<T>,
        biases: &TokenTensor2D<i32>,
        output: &TokenTensor2D<T>,
    ) -> (TokenBuffer2D<f32>, f32, TokenBuffer2D<i32>, i32) {
        (
            TokenBuffer2D::from(
                biases.scale[0] / output.scale[0]
                    * biases
                        .buffer
                        .add_scalar(-biases.zero_point[0])
                        .cast::<f32>(),
            ),
            input.scale[0] * weights.scale[0] / output.scale[0],
            TokenBuffer2D::from(DMatrix::from_rows(&[
                convert_ref::<DMatrix<T>, DMatrix<i32>>(&weights.buffer).row_sum()
                    * i32::from_subset(&input.zero_point[0]),
            ])),
            input.shape[1] as i32
                * i32::from_subset(&input.zero_point[0])
                * i32::from_subset(&weights.zero_point[0]),
        )
    }
    fn zero_constants(
        weights: &TokenTensor2D<T>,
        biases: &TokenTensor2D<i32>,
    ) -> (TokenBuffer2D<f32>, f32, TokenBuffer2D<i32>, i32) {
        (
            TokenBuffer2D::from(DMatrix::from_fn(
                biases.shape[0],
                biases.shape[1],
                |_, _| 0f32,
            )),
            0f32,
            TokenBuffer2D::from(DMatrix::from_fn(1, weights.shape[1], |_, _| 0i32)),
            0i32,
        )
    }
}
impl<T: TokenQuantized> TrainToTokens for TokenFullyConnected<T> {
    fn add_attrs(&self, attrs: &mut ItemStruct) {
        let filters_ident = format_ident!("weights{}", self.layer_index as usize);
        let filters_gradient_ident = format_ident!("weights{}_gradient", self.layer_index as usize);
        let filters_type = self.weights.type_tokens();
        let dim_weights_0 = self.weights.shape[0];
        let dim_weights_1 = self.weights.shape[1];
        let constants_field_name = format_ident!("constants{}", self.layer_index as usize);
        let constants_gradient_field_name =
            format_ident!("constants{}_gradient", self.layer_index as usize);
        let (constants_0, _, constants_2, _) = &self.constants;
        let dim00 = constants_0.shape().0;
        let dim01 = constants_0.shape().1;
        let dim10 = constants_2.shape().0;
        let dim11 = constants_2.shape().1;
        let constants_field_type: syn::Type = parse_quote! {
            (SMatrix<f32,#dim00,#dim01>,f32,SMatrix<i32,#dim10,#dim11>,i32)
        };

        let constants_field: syn::Field = syn::parse_quote! {
            #constants_field_name: #constants_field_type
        };
        let constants_gradient_field: syn::Field = syn::parse_quote! {
            #constants_gradient_field_name: #constants_field_type
        };
        let filters_field: syn::Field = syn::parse_quote! {
            #filters_ident: #filters_type
        };
        let filters_gradient_field: syn::Field = syn::parse_quote! {
            #filters_gradient_ident: nalgebra::SMatrix<i32,#dim_weights_0,#dim_weights_1>
        };
        match &mut attrs.fields {
            syn::Fields::Named(ref mut fields_named) => {
                fields_named.named.push(constants_field);
                fields_named.named.push(filters_field);
                fields_named.named.push(constants_gradient_field);
                fields_named.named.push(filters_gradient_field);
            }
            _ => panic!("add_fields only works with structs with named fields"),
        }
    }
    fn define_members(&self, declarations: &mut TokenStream2) {
        let constants_field_name = format_ident!("constants{}", self.layer_index as usize);
        let filters_ident = format_ident!("weights{}", self.layer_index as usize);
        let constants_gradient_field_name =
            format_ident!("constants{}_gradient", self.layer_index as usize);
        let filters_gradient_ident = format_ident!("weights{}_gradient", self.layer_index as usize);
        let filters = &self.weights;
        let (constants_0, constants_1, constants_2, constants_3) = &self.constants;
        let (
            constants_gradient_0,
            constants_gradient_1,
            constants_gradient_2,
            constants_gradient_3,
        ) = &self.constants_gradient;
        let ts = quote! {
            #filters_ident : #filters,
            #filters_gradient_ident : SMatrix::from_fn(|_,_|0i32),
            #constants_field_name : (#constants_0, #constants_1, #constants_2, #constants_3),
            #constants_gradient_field_name : (#constants_gradient_0, #constants_gradient_1, #constants_gradient_2, #constants_gradient_3),
        };
        ts.to_tokens(declarations);
    }
    fn train_ops(&self, backward: &mut TokenStream2) {
        let weights_ident: syn::Expr = {
            let field_ident = format_ident!("weights{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let weights_gradient_ident: syn::Expr = {
            let field_ident = format_ident!("weights{}_gradient", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let output_ident = if self.layer_index >= 0 {
            format_ident!("input{}", self.layer_index as usize)
        } else {
            format_ident!("input")
        };
        let input_ident = if self.layer_index > 0 {
            format_ident!("input{}", (self.layer_index - 1) as usize)
        } else {
            format_ident!("input")
        };
        let activation = self.fused_activation;
        let bias_scale = self.scale_bias;
        let constants_ident: syn::Expr = {
            let field_ident = format_ident!("constants{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let constants_gradient_ident: syn::Expr = {
            let field_ident = format_ident!("constants{}_gradient", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let back_norm = self.back_norm;
        if !self.reshape {
            let prepend = quote! {
                let backward_gradient = microflow::gradient_fully_connected::update_grad_fully_connected(
                    &#input_ident,
                    & #output_ident,
                    & #weights_ident,
                    &mut #weights_gradient_ident,
                    & #constants_ident,
                    &mut #constants_gradient_ident,
                    #activation,
                    backward_gradient,
                    #bias_scale,
                    learning_rate,
                    #back_norm,
                );
                // println!("input: {}, {}, {}", #input_ident.buffer.view((0, 0), (1, 4)),#input_ident.zero_point[0], #input_ident.scale[0]);
                // println!("output net: {}",#output_ident.buffer);
                // println!("input: {}",#input_ident.buffer);
                // println!("input_zero_point: {}",#input_ident.zero_point[0]);
                // println!("weights_gradient: {}",#weights_gradient_ident);
            };
            let mut ts = TokenStream2::new();
            prepend.to_tokens(&mut ts);
            ts.extend(backward.clone());
            *backward = ts;
        } else if let Some(input) = &self.input {
            let shape_0 = input[0];
            let shape_1 = input[1];
            let shape_2 = input[2];
            let shape_3 = input[3];
            let prepend = quote! {
                let backward_gradient :  Tensor4D<i32, #shape_0, #shape_1, #shape_2, #shape_3, 1>= Tensor2D{
                    buffer: microflow::gradient_fully_connected::update_grad_fully_connected(
                    &#input_ident,
                    & #output_ident,
                    & #weights_ident,
                    &mut #weights_gradient_ident,
                    & #constants_ident,
                    &mut #constants_gradient_ident,
                    #activation,
                    backward_gradient,
                    #bias_scale,
                    learning_rate,
                    #back_norm,
                    ),
                    zero_point: [0i32],
                    scale: [1f32]
                    }.into();
                let backward_gradient = backward_gradient.buffer;
                let #input_ident = #input_ident .into();
                // println!("input: {}, {}, {}", #input_ident.buffer.view((0, 0), (1, 4)),#input_ident.zero_point[0], #input_ident.scale[0]);
                // println!("output net: {}",#output_ident.buffer);
                // println!("input: {}",#input_ident.buffer);
                // println!("input_zero_point: {}",#input_ident.zero_point[0]);
                // println!("weights_gradient: {}",#weights_gradient_ident);
            };
            let mut ts = TokenStream2::new();
            prepend.to_tokens(&mut ts);
            ts.extend(backward.clone());
            *backward = ts;
        }
    }
    fn switch_train(&mut self) {
        self.train = !self.train;
    }

    fn update_ops(&self, updates: &mut TokenStream2) {
        let weights_ident: syn::Expr = {
            let field_ident = format_ident!("weights{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let weights_gradient_ident: syn::Expr = {
            let field_ident = format_ident!("weights{}_gradient", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let constants_ident: syn::Expr = {
            let field_ident = format_ident!("constants{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let constants_gradient_ident: syn::Expr = {
            let field_ident = format_ident!("constants{}_gradient", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let weights_size = self.weights.shape.iter().fold(1, |acc, el| acc * el);
        let weights_shape_0 = self.weights.shape[0];
        let weights_shape_1 = self.weights.shape[1];
        let input_zero_point = self.input_zero_point;
        let perc: usize = (weights_size as f32 * 0.25).floor() as usize;
        let clip_norm = self.gradient_norm;
        let update = quote! {
            // microflow::update_layer::update_weights_2D(
            //     &mut #weights_ident,
            //     &#weights_gradient_ident,
            //     batch_size,
            //     learning_rate,
            // );
            // microflow::update_layer::update_weights_max_2D(
            //     &mut #weights_ident,
            //     &#weights_gradient_ident,
            //     batch_size,
            //     learning_rate,
            // );
            microflow::update_layer::update_weights_clip_norm_2D(
                &mut #weights_ident,
                &#weights_gradient_ident,
                batch_size,
                learning_rate,
                #clip_norm,
            );
            microflow::update_layer::update_weights_2D_float(
                &mut #constants_ident.0,
                &#constants_gradient_ident.0,
                batch_size,
                learning_rate,
            );
            microflow::update_layer::update_constants_fully_connected(
                &#weights_ident,
                &mut #constants_ident,
                #input_zero_point
            );
            // println!("gradient bias:{}",#constants_gradient_ident.0[0]);
            // println!("bias:{}",#constants_ident.0[0]);
            #weights_gradient_ident = nalgebra::SMatrix::zeros();
            #constants_gradient_ident.0 = nalgebra::SMatrix::zeros();
            // println!("gradient weights: {}",weight_gradient.iter().fold(String::new(), |accarr, batch|accarr + &batch.map(|el|el.iter().fold(String::new(),|sum, el1|sum +" "+ &el1.to_string())).to_string()));
            // println!("gradient input: {}",backward_gradient[0].map(|el|el.iter().fold(String::new(),|sum, el1|sum +" " +&el1.to_string())));
            // println!("mean gradient conv: {}",weight_gradient[0].map(|el|el.iter().fold(0f32,|sum, el1|sum+(*el1 as f32).abs() / el.len() as f32)).mean());
        };
        update.to_tokens(updates);
    }
}

impl<T: TokenQuantized> ToTokens for TokenFullyConnected<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let reshape = if self.reshape {
            quote!(.into())
        } else {
            quote!()
        };
        let weights_ident: syn::Expr = if self.layer_index >= 0 {
            let field_ident = format_ident!("weights{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        } else {
            let field_ident = format_ident!("weights{}", self.index);
            parse_quote!(#field_ident)
        };
        let constants_ident: syn::Expr = if self.layer_index >= 0 {
            let field_ident = format_ident!("constants{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        } else {
            let (constants_0, constants_1, constants_2, constants_3) = &self.constants;
            parse_quote!((#constants_0, #constants_1, #constants_2, #constants_3))
        };
        let weights_type = self.weights.type_tokens();
        let weights = &self.weights;
        let weights_declaration = if self.layer_index < 0 {
            quote! {const #weights_ident: #weights_type = #weights;}
        } else {
            quote! {}
        };
        let output_shape = &self.output.shape;
        let output_scale = self.output.scale[0];
        let output_zero_point = self.output.zero_point[0];
        let fused_activation = self.fused_activation;
        let reference_tok = if self.layer_index >= 0 && self.train {
            quote! {&}
        } else {
            quote! {}
        };
        let output_name = if self.layer_index >= 0 && self.train {
            format_ident!("input{}", self.layer_index as usize)
        } else {
            format_ident!("input")
        };
        let input_name = if self.layer_index > 0 && self.train {
            format_ident!("input{}", (self.layer_index - 1) as usize)
        } else {
            format_ident!("input")
        };
        let func_name: syn::Path = if self.layer_index >= 0 && self.train {
            parse_quote!(microflow::ops::fully_connected_borrow)
        } else {
            parse_quote!(microflow::ops::fully_connected)
        };

        let ts = quote! {
            #weights_declaration
            let #input_name = #input_name #reshape;
            let #output_name: microflow::tensor::Tensor2D<_, #(#output_shape),*, 1usize> =
                #func_name(
                    #reference_tok #input_name,
                    & #weights_ident,
                    [#output_scale],
                    [#output_zero_point],
                    microflow::ops::FullyConnectedOptions {
                        fused_activation: #fused_activation,
                    },
                    #constants_ident,
            );
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::dmatrix;

    use super::*;

    fn setup() -> TokenFullyConnected<i8> {
        TokenFullyConnected {
            weights: TokenTensor2D {
                buffer: TokenBuffer2D::from(dmatrix![
                    1, 2, 3;
                    4, 5, 6
                ]),
                shape: vec![2, 3],
                scale: vec![0.7],
                zero_point: vec![8],
            },
            output: TokenTensor2D {
                buffer: TokenBuffer2D::new(),
                shape: vec![1, 3],
                scale: vec![0.9],
                zero_point: vec![10],
            },
            fused_activation: TokenFusedActivation::Relu,
            constants: (
                TokenBuffer2D::from(dmatrix![11., 12.]),
                13.,
                TokenBuffer2D::from(dmatrix![14, 15]),
                16,
            ),
            index: 0,
            reshape: false,
            layer_index: -1,
            train: false,
            scale_bias: 1f32,
        }
    }

    #[test]
    fn fully_connected_preprocess() {
        let layer = setup();
        let input = TokenTensor2D {
            buffer: TokenBuffer2D::new(),
            shape: vec![1, 2],
            scale: vec![0.17],
            zero_point: vec![18],
        };
        let biases = TokenTensor2D {
            buffer: TokenBuffer2D::from(dmatrix![
                19;
                20;
                21
            ]),
            shape: vec![3, 1],
            scale: vec![0.22],
            zero_point: vec![23],
        };
        let constants =
            TokenFullyConnected::preprocess(&input, &layer.weights, &biases, &layer.output);
        assert_eq!(
            constants.0 .0,
            Some(dmatrix![-0.9777778; -0.73333335; -0.4888889])
        );
        assert_eq!(constants.1, 0.13222224);
        assert_eq!(constants.2 .0, Some(dmatrix![90, 126, 162]));
        assert_eq!(constants.3, 288);
    }

    #[test]
    fn fully_connected_to_tokens() {
        let layer = setup();
        let weights = &layer.weights;
        let fused_activation = layer.fused_activation;
        let constants_0 = &layer.constants.0;
        let constants_2 = &layer.constants.2;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                const weights_0: microflow::tensor::Tensor2D<i8, 2usize, 3usize, 1usize> = #weights;
                let input: microflow::tensor::Tensor2D<_, 1usize, 3usize, 1usize> =
                    microflow::ops::fully_connected(
                        input,
                        &weights_0,
                        [0.9f32],
                        [10i8],
                        microflow::ops::FullyConnectedOptions {
                            fused_activation: #fused_activation,
                        },
                        (#constants_0, 13f32, #constants_2, 16i32)
                );
            }
            .to_string()
        );
    }
}
