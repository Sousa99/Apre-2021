use nalgebra::{DVector, DMatrix};

pub type Element = f64;
pub type Vector = DVector<Element>;
pub type Matrix = DMatrix<Element>;

#[allow(dead_code)]
#[derive(Copy, Clone)]
enum IterationType {
    Stochastic,
}

#[allow(dead_code)]
#[derive(Copy, Clone)]
enum ErrorFunctionType {
    HalfSumSquaredError,
}

#[allow(dead_code)]
#[derive(Copy, Clone)]
enum ActivationFunctionType {
    Sigmoid
}

// ================================== START DEFINE PARAMETERS ==================================

const ERROR_FUNCTION : ErrorFunctionType = ErrorFunctionType::HalfSumSquaredError;

fn get_layers() -> Vec<Layer> {
    return vec![
        build_layer((3, 5), 0.1, 0.0, ActivationFunctionType::Sigmoid),
        build_layer((2, 3), 0.1, 0.0, ActivationFunctionType::Sigmoid),
    ];
}

// ================================== END DEFINE PARAMETERS ==================================

#[derive(Clone)]
pub struct PointInfo {
    point: Vector,
    target: Vector,
}

pub fn build_point_info(point: Vector, target: Vector) -> PointInfo {
    let point_info : PointInfo = PointInfo {
        point: point,
        target: target,
    };

    return point_info;
}

struct Layer {
    weights: Matrix,
    bias: Vector,
    function: ActivationFunctionType,
}

fn build_layer(dimension: (usize, usize), weight_value: Element, bias_value: Element, function: ActivationFunctionType) -> Layer {
    let layer : Layer = Layer {
        weights: Matrix::from_element(dimension.0, dimension.1, weight_value),
        bias: Vector::from_element(dimension.0, bias_value),
        function: function,
    };

    return layer;
}

pub struct Network {
    layers: Vec<Layer>,
    points:  Vec<PointInfo>,
    learning_rate: f64,
    iteration_type: IterationType,
    error_function: ErrorFunctionType,
}

impl Network {

    pub fn do_iteration(&mut self) {

        match self.iteration_type {
            IterationType::Stochastic => {
                let copy_points = self.points.clone();
                for (point_index, point_info) in copy_points.iter().enumerate() {
        
                    println!("for point {}", point_index + 1);
                    println!("----------------------------------------");
                    let (xs, zs) = self.forward_propagation(&point_info.point);
                    println!();
                    self.backward_propagation(xs, zs, point_info.target.clone());
                    println!();
                }
            }
        }
    }

    pub fn run_point(&self, point: &Vector) {
        self.forward_propagation(point);
    }

    fn forward_propagation(&self, point: &Vector) -> (Vec<Vector>, Vec<Vector>) {

        println!("FORWARD PROPAGATION");
        let mut xs : Vec<Vector> = vec![point.clone()];
        let mut zs : Vec<Vector> = Vec::new();

        for layer in self.layers.iter() {
            zs.push(compute_z(&layer.weights, xs.last().unwrap(), &layer.bias));
            xs.push(compute_x(zs.last().unwrap(), layer.function));
        }

        // ============= PRINT INFORMATION =============
        println!("X^[0] (transposed) = {}", xs[0].transpose());
        for index in 0..self.layers.len() {
            println!("Z^[{}] (transposed) = {}", index + 1, zs[index].transpose());
            println!("X^[{}] (transposed) = {}", index + 1, xs[index + 1].transpose());
        }
        println!();

        return (xs, zs);
    }

    fn backward_propagation(&mut self, xs: Vec<Vector>, zs: Vec<Vector>, target: Vector) {

        println!("BACKWARD PROPAGATION");
        let deltas : Vec<Vector> = self.compute_deltas(&xs, &zs, target);
        self.print_backward_propagation(&deltas);

        self.update(deltas, &xs);
    }

    fn compute_deltas(&self, xs: &Vec<Vector>, zs: &Vec<Vector>, target: Vector) -> Vec<Vector> {
        let mut deltas_reversed : Vec<Vector> = Vec::new();

        for (layer_index, layer) in self.layers.iter().rev().enumerate() {
            let rev_layer_index = self.layers.len() - layer_index;

            if layer_index == 0 {
                let delta : Vector = match (self.error_function, layer.function) {
                    (ErrorFunctionType::HalfSumSquaredError, ActivationFunctionType::Sigmoid) => {
                        let sigmoid_vector = apply_function_to_vector(zs.get(rev_layer_index - 1).unwrap(), sigmoid);
                        let dimension = sigmoid_vector.shape().0;
                        let difference_one = Vector::from_element(dimension, 1.0) - sigmoid_vector.clone();
                        let sigmoid_derivative = sigmoid_vector.component_mul(&difference_one);

                        (xs.get(rev_layer_index).unwrap() - target.clone()).component_mul(&sigmoid_derivative)
                    }
                };

                deltas_reversed.push(delta);

            } else {
                let delta : Vector = match layer.function {
                    ActivationFunctionType::Sigmoid => {
                        let last_delta = deltas_reversed.last().unwrap();

                        let sigmoid_vector = apply_function_to_vector(zs.get(rev_layer_index - 1).unwrap(), sigmoid);
                        let dimension = sigmoid_vector.shape().0;
                        let difference_one = Vector::from_element(dimension, 1.0) - sigmoid_vector.clone();
                        let sigmoid_derivative = sigmoid_vector.component_mul(&difference_one);

                        let weight = &self.layers.get(rev_layer_index).unwrap().weights;

                        (weight.transpose() * last_delta).component_mul(&sigmoid_derivative)
                    }
                };

                deltas_reversed.push(delta);
            }
        }

        deltas_reversed.reverse();
        return deltas_reversed;
    }

    fn update(&mut self, deltas: Vec<Vector>, xs: &Vec<Vector>) {

        println!("(update)");

        let mut new_layers: Vec<Layer> = Vec::new();
        for (index, ((layer, delta), x)) in self.layers.iter().zip(deltas.iter()).zip(xs.iter()).enumerate() {

            let new_weight_variation = delta * x.transpose();
            let new_weight = layer.weights.clone() - self.learning_rate * new_weight_variation.clone();

            let new_bias_variation = delta * 1.0;
            let new_bias = layer.bias.clone() - self.learning_rate * new_bias_variation.clone();

            // ============ PRINT VALUES ============

            println!("dE / dW^[{}] = delta^[{}] . (dz^[{}] / dW[{}])^T = {}", index + 1, index + 1, index + 1, index + 1, new_weight_variation);
            println!("          W^[{}] = W^[{}] - lr * dE / dW^[{}] = {}", index + 1, index + 1, index + 1, new_weight);

            println!("dE / db^[{}] = delta^[{}] . (dz^[{}] / db[{}])^T = {}", index + 1, index + 1, index + 1, index + 1, new_bias_variation);
            println!("          b^[{}] = b^[{}] - lr * dE / db^[{}] = {}", index + 1, index + 1, index + 1, new_bias);

            println!();

            // ============ STORE NEW LAYER ============

            let new_layer = Layer {
                weights: new_weight,
                bias: new_bias,
                function: layer.function,
            };
            new_layers.push(new_layer);
        }

        self.layers = new_layers;
    }

    pub fn print_schema(&self) {

        for (layer_index, layer) in self.layers.iter().enumerate() {
            
            println!("W^[{}] (Weight Matrix) = {}", layer_index + 1, layer.weights);
            println!("b^[{}] (Bias Vector) = {}", layer_index + 1, layer.bias);
            println!();
        }
    }

    fn print_backward_propagation(&self, deltas: &Vec<Vector>) {

        match self.iteration_type {
            IterationType::Stochastic => {

                // Print error function
                print!("E (t, x^[{}]) = ", self.layers.len());
                match self.error_function {
                    ErrorFunctionType::HalfSumSquaredError => println!("1/2 ( x^[{}] - t )^2", self.layers.len()),
                }
                println!();

                // Print derivatives
                println!("(calculate derivatives)");

                // Error function
                print!("dE (t, x^[{}]) / dx^[{}] = ", self.layers.len(), self.layers.len());
                match self.error_function {
                    ErrorFunctionType::HalfSumSquaredError => println!("( x^[{}] - t )", self.layers.len()),
                }
                println!();

                // Intermediate steps
                for (layer_index, layer) in self.layers.iter().rev().enumerate() {
                    let rev_layer_index = self.layers.len() - layer_index;

                    print!("dx^[{}] (z^[{}]) / dz^[{}] = ", rev_layer_index, rev_layer_index, rev_layer_index);
                    match layer.function {
                        ActivationFunctionType::Sigmoid => println!("sigmoid(z^[{}]) (1 - sigmoid(z^[{}]))", rev_layer_index, rev_layer_index),
                    }

                    print!("dz^[{}] (W^[{}], b^[{}], x^[{}]) / dW^[{}] = ", rev_layer_index, rev_layer_index, rev_layer_index, rev_layer_index - 1, rev_layer_index);
                    println!("x^[{}]", rev_layer_index - 1);
                    
                    print!("dz^[{}] (W^[{}], b^[{}], x^[{}]) / db^[{}] = ", rev_layer_index, rev_layer_index, rev_layer_index, rev_layer_index - 1, rev_layer_index);
                    println!("1");

                    print!("dz^[{}] (W^[{}], b^[{}], x^[{}]) / dx^[{}] = ", rev_layer_index, rev_layer_index, rev_layer_index, rev_layer_index - 1, rev_layer_index - 1);
                    println!("W^[{}]", rev_layer_index);

                    println!();
                }

                println!("(calculate deltas)");

                // Deltas
                for (layer_index, (layer, delta)) in self.layers.iter().rev().zip(deltas.iter().rev()).enumerate() {
                    let rev_layer_index = self.layers.len() - layer_index;

                    print!("delta^[{}] = ", rev_layer_index);
                    if layer_index == 0 {
                        print!("dE / dx^[{}] o dx^[{}] / dz^[{}] = ", rev_layer_index, rev_layer_index, rev_layer_index);
                        match (self.error_function, layer.function) {
                            (ErrorFunctionType::HalfSumSquaredError, ActivationFunctionType::Sigmoid) => print!("( x^[{}] - t ) o sigmoid(z^[{}]) (1 - sigmoid(z^[{}])) = ", rev_layer_index, rev_layer_index, rev_layer_index),
                        }
                        println!("(transposed) {}", delta.transpose());
                        
                    } else {
                        print!("( dz^[{}] / dx^[{}] ) ^T . delta^[{}] o dx^[{}] / dz^[{}] = ", rev_layer_index + 1, rev_layer_index, rev_layer_index + 1, rev_layer_index, rev_layer_index);
                        match layer.function {
                            ActivationFunctionType::Sigmoid => print!("(W^[{}])^T . delta^[{}] o sigmoid(z^[{}]) (1 - sigmoid(z^[{}])) = ", rev_layer_index, rev_layer_index, rev_layer_index - 1, rev_layer_index - 1),
                        }
                        println!("(transposed) {}", delta.transpose());
                    }
    
                }
            },
        }
    }
}

pub fn build_network(points: Vec<PointInfo>, learning_rate: f64) -> Network {
    let network : Network = Network {
        points: points,
        learning_rate: learning_rate,
        iteration_type: IterationType::Stochastic,
        error_function: ERROR_FUNCTION,
        layers: get_layers(),
    };

    return network;
}

// ================================ AUXILIARY METHODS ================================

fn compute_z(weights: &Matrix, point: &Vector, bias: &Vector) -> Vector {
    let z : Vector = weights * point + bias;
    return z;
}

fn compute_x(z: &Vector, function_type: ActivationFunctionType) -> Vector {
    return match function_type {
        ActivationFunctionType::Sigmoid => apply_function_to_vector(z, sigmoid),
    } 
}

fn sigmoid(value: Element) -> Element {
    let denominator = 1.0 + (- value).exp();
    return 1.0 / denominator;
}

fn apply_function_to_vector(vector: &Vector, function: fn(Element) -> Element) -> Vector {

    let mut values : Vec<Element> = Vec::new();
    for &value in vector.iter() { values.push(function(value)) }

    let new_vector : Vector = Vector::from_vec(values);
    return new_vector;
}