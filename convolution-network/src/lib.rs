use nalgebra::{DMatrix};

pub type Element = f64;
pub type Matrix = DMatrix<Element>;

#[derive(Copy, Clone)]
enum OperationType {
    Convolutional,
    MaxPooling
}

struct Layer {
    operation: OperationType,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    kernels: Option<Vec<Matrix>>,
}

// ============================= DEFINE NETWORK =============================

fn build_network() -> Vec<Layer> {
    return vec![
        Layer {
            operation: OperationType::Convolutional,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            kernels: Some(vec![
                Matrix::from_vec(3, 3, vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
                Matrix::from_vec(3, 3, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            ])
        },

        Layer {
            operation: OperationType::MaxPooling,
            kernel_size: (2, 2),
            stride: (2, 2),
            padding: (0, 0),
            kernels: None
        },
    ];
}

pub struct ConvolutionNetwork {
    layers: Vec<Layer>,
}

impl ConvolutionNetwork {

    pub fn run_input(&self, input: Vec<Matrix>) {

        let mut current_input: Vec<Matrix> = input;
        for (index, layer) in self.layers.iter().enumerate() {

            let mut output: Vec<Matrix> = Vec::new();
            let mut channels : usize = current_input.len();
            if layer.kernels.is_some() { channels = layer.kernels.as_ref().unwrap().len() }
            let output_dimension = calculate_output_dimension(current_input[0].shape(), layer.kernel_size, layer.padding, layer.stride, channels);

            let mut kernels : Vec<Option<&Matrix>> = Vec::new();
            match &layer.kernels {
                Some(some) => for sub_kernel in some { kernels.push(Some(&sub_kernel)) }
                None => kernels.push(None),
            }

            for kernel in kernels {

                let mut computed_output : Matrix = Matrix::from_element(output_dimension.0, output_dimension.1, 0.0);
                for input in current_input.iter() {

                    let padded_input = add_padding(input, layer.padding);
                    let tmp_computed_output = compute_output(&padded_input, layer.operation, layer.kernel_size, kernel, layer.stride, output_dimension);
                    
                    match layer.operation {
                        OperationType::Convolutional => computed_output = computed_output + tmp_computed_output,
                        OperationType::MaxPooling => output.push(tmp_computed_output),
                    }
                }

                match layer.operation {
                    OperationType::Convolutional => output.push(computed_output),
                    OperationType::MaxPooling => (),
                }
            }

            println!("==================================== Layer {} ====================================", index + 1);
            println!("Computed Output Shape = {:?}", output_dimension);

            for output_matrix in output.iter() { println!("{}", output_matrix) }


            current_input = output;
        }
    }
}

pub fn build_convolution_network() -> ConvolutionNetwork {
    let convolution_network: ConvolutionNetwork = ConvolutionNetwork {
        layers: build_network(),
    };

    return convolution_network;
}

// ============================= AUXILIARY METHODS =============================
pub fn calculate_output_dimension(input: (usize, usize), kernel: (usize, usize), padding: (usize, usize), stride: (usize, usize), channels: usize) -> (usize, usize, usize) {

    let output_vertical: usize = calculate_output_single_dimension(input.0, kernel.0, padding.0, stride.0);
    let output_horizontal: usize = calculate_output_single_dimension(input.1, kernel.1, padding.1, stride.1);

    return (output_vertical, output_horizontal, channels);
}

pub fn calculate_output_single_dimension(input: usize, kernel: usize, padding: usize, stride: usize) -> usize {
    return (input - kernel + 2 * padding) / stride + 1;
}

fn add_padding(matrix: &Matrix, padding: (usize, usize)) -> Matrix {

    let dimension = matrix.shape();
    let mut padded_matrix : Matrix = matrix.clone();
    // Add padding Vertical
    padded_matrix = padded_matrix.insert_rows(dimension.0, padding.0, 0.0);
    padded_matrix = padded_matrix.insert_rows(0, padding.0, 0.0);
    // Add padding Horizontal
    padded_matrix = padded_matrix.insert_columns(dimension.1, padding.1, 0.0);
    padded_matrix = padded_matrix.insert_columns(0, padding.1, 0.0);

    return padded_matrix;
}

fn compute_output(matrix: &Matrix, operation: OperationType, kernel_size: (usize, usize), kernel: Option<&Matrix>, stride: (usize, usize), output_shape: (usize, usize, usize)) -> Matrix {

    let matrix_size: (usize, usize) = matrix.shape();
    let mut position: (usize, usize) = (0, 0);

    let mut values_vec : Vec<Element> = Vec::new();
    while position.1 + kernel_size.1 <= matrix_size.1 {

        // Compute value from this position
        let mut values_used : Vec<Element> = Vec::new();
        for variation_h in 0..kernel_size.0 {
            for variation_w in 0..kernel_size.1 {
                let tmp_position = (position.0 + variation_h, position.1 + variation_w);
                values_used.push(matrix[tmp_position]);
            }
        }

        let output_value : Element = do_operation(values_used, operation, kernel);
        values_vec.push(output_value);

        position.0 = position.0 + stride.0;
        if position.0 + kernel_size.0 > matrix_size.0 {
            position.0 = 0;
            position.1 = position.1 + stride.1;
        }
    }

    let output : Matrix = Matrix::from_vec(output_shape.0, output_shape.1, values_vec);
    return output;
}

fn do_operation(values: Vec<Element>, operation: OperationType, kernel_option: Option<&Matrix>) -> Element {
    let output : Element = match (operation, kernel_option) {
        (OperationType::Convolutional, Some(kernel)) => {
            let mut sum : Element = 0.0;
            for (value, kernel_value) in values.iter().zip(kernel.iter()) { sum = sum + value * kernel_value }
            
            sum
        },
        (OperationType::MaxPooling, None) => {
            let mut maximum : Element = - std::f64::INFINITY;
            for &value in values.iter() {
                if value > maximum { maximum = value }
            }

            maximum
        }
        
        (OperationType::Convolutional, None) => panic!("Convolutional must have a kernel!"),
        (OperationType::MaxPooling, Some(_)) => panic!("Pooling should not have kernels!"),
    };

    return output;
}