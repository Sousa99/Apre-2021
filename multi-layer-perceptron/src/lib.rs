use nalgebra::{DVector, DMatrix};

pub type Element = f64;
pub type Vector = DVector<Element>;
pub type Matrix = DMatrix<Element>;

#[allow(dead_code)]
#[derive(Copy, Clone)]
enum IterationType {
    Stochastic,
    Predefined,
}

#[allow(dead_code)]
#[derive(Copy, Clone)]
enum ErrorFunctionType {
    HalfSumSquaredError,
}

#[allow(dead_code)]
#[derive(Copy, Clone)]
enum ActivationFunctionType {
    Sigmoid,
    HyperbolicTangent,
}

// ================================== START DEFINE PARAMETERS ==================================

const ITERATION : IterationType = IterationType::Stochastic;
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
                    let deltas = self.backward_propagation(&xs, &zs, point_info.target.clone());
                    self.update_stochastic(deltas, &xs);
                    println!();
                }
            },
            
            IterationType::Predefined => {

                let mut all_xs: Vec<Vec<Vector>> = Vec::new();
                let mut all_zs: Vec<Vec<Vector>> = Vec::new();
                let mut all_deltas: Vec<Vec<Vector>> = Vec::new();
                let copy_points = self.points.clone();

                for (point_index, point_info) in copy_points.iter().enumerate() {
                    println!("for point {}", point_index + 1);
                    println!("----------------------------------------");
                    let (xs, zs) = self.forward_propagation(&point_info.point);

                    all_xs.push(xs);
                    all_zs.push(zs);
                }

                println!();
                for (point_index, (point_info, (xs, zs))) in copy_points.iter().zip(all_xs.iter().zip(all_zs.iter())).enumerate() {
                    println!("for point {}", point_index + 1);
                    println!("----------------------------------------");
                    let deltas = self.backward_propagation(&xs, &zs, point_info.target.clone());
                    all_deltas.push(deltas);
                }

                println!();
                self.update_predefined(all_deltas, &mut all_xs);
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
        println!("X^[0] = (transposed) {}", xs[0].transpose());
        for index in 0..self.layers.len() {
            println!("Z^[{}] = (transposed) {}", index + 1, zs[index].transpose());
            println!("X^[{}] = (transposed) {}", index + 1, xs[index + 1].transpose());
        }
        println!();

        return (xs, zs);
    }

    fn backward_propagation(&mut self, xs: &Vec<Vector>, zs: &Vec<Vector>, target: Vector) -> Vec<Vector> {

        println!("BACKWARD PROPAGATION");
        let deltas : Vec<Vector> = self.compute_deltas(xs, zs, target);
        self.print_backward_propagation(&deltas);
        return deltas;
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
                    },

                    (ErrorFunctionType::HalfSumSquaredError, ActivationFunctionType::HyperbolicTangent) => {
                        let hyperbolic_tangent_vector = apply_function_to_vector(zs.get(rev_layer_index - 1).unwrap(), hyperbolic_tangent);
                        let squared_tangent = apply_function_to_vector(&hyperbolic_tangent_vector, square);
                        let dimension = squared_tangent.shape().0;
                        let difference_one = Vector::from_element(dimension, 1.0) - squared_tangent.clone();

                        (xs.get(rev_layer_index).unwrap() - target.clone()).component_mul(&difference_one)
                    },
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
                    },

                    ActivationFunctionType::HyperbolicTangent => {
                        let last_delta = deltas_reversed.last().unwrap();

                        let hyperbolic_tangent_vector = apply_function_to_vector(zs.get(rev_layer_index - 1).unwrap(), hyperbolic_tangent);
                        let squared_tangent = apply_function_to_vector(&hyperbolic_tangent_vector, square);
                        let dimension = squared_tangent.shape().0;
                        let difference_one = Vector::from_element(dimension, 1.0) - squared_tangent.clone();

                        let weight = &self.layers.get(rev_layer_index).unwrap().weights;

                        (weight.transpose() * last_delta).component_mul(&difference_one)
                    },
                };

                deltas_reversed.push(delta);
            }
        }

        deltas_reversed.reverse();
        return deltas_reversed;
    }

    fn update_stochastic(&mut self, deltas: Vec<Vector>, xs: &Vec<Vector>) {

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

            println!("dE / db^[{}] = delta^[{}] . (dz^[{}] / db[{}])^T = (transposed) {}", index + 1, index + 1, index + 1, index + 1, new_bias_variation.transpose());
            println!("          b^[{}] = b^[{}] - lr * dE / db^[{}] = (transposed) {}", index + 1, index + 1, index + 1, new_bias.transpose());

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

    fn update_predefined(&mut self, mut all_deltas: Vec<Vec<Vector>>, all_xs: &mut Vec<Vec<Vector>>) {

        // Fix deltas and xs
        let mut fixed_deltas: Vec<Vec<Vector>> = Vec::new();
        let mut fixed_xs: Vec<Vec<Vector>> = Vec::new();
        for index in 0..self.layers.len() {

            let mut tmp_deltas: Vec<Vector> = Vec::new();
            for point_deltas in all_deltas.iter_mut() { tmp_deltas.push(point_deltas.remove(0)) }
            fixed_deltas.push(tmp_deltas);
            
            let mut tmp_xs: Vec<Vector> = Vec::new();
            for point_xs in all_xs.iter_mut() { tmp_xs.push(point_xs.remove(0)) }
            fixed_xs.push(tmp_xs);
        }

        println!("(update)");

        let mut new_layers: Vec<Layer> = Vec::new();
        for (index, ((layer, deltas), xs)) in self.layers.iter().zip(fixed_deltas.iter()).zip(fixed_xs.iter()).enumerate() {

            let weight_shape = layer.weights.shape();
            let mut new_weight_variation = Matrix::from_element(weight_shape.0, weight_shape.1, 0.0);
            for (delta, x) in deltas.iter().zip(xs.iter()) { new_weight_variation = new_weight_variation + delta * x.transpose() }
            let new_weight = layer.weights.clone() - self.learning_rate * new_weight_variation.clone();

            let bias_shape = layer.bias.shape();
            let mut new_bias_variation = Vector::from_element(bias_shape.0, 0.0);
            for delta in deltas.iter() { new_bias_variation = new_bias_variation + delta * 1.0 }
            let new_bias = layer.bias.clone() - self.learning_rate * new_bias_variation.clone();

            // ============ PRINT VALUES ============

            println!("dE / dW^[{}] = Σ delta^[{}] . (dz^[{}] / dW[{}])^T = {}", index + 1, index + 1, index + 1, index + 1, new_weight_variation);
            println!("          W^[{}] = W^[{}] - lr * dE / dW^[{}] = {}", index + 1, index + 1, index + 1, new_weight);

            println!("dE / db^[{}] = Σ delta^[{}] . (dz^[{}] / db[{}])^T = (transposed) {}", index + 1, index + 1, index + 1, index + 1, new_bias_variation.transpose());
            println!("          b^[{}] = b^[{}] - lr * dE / db^[{}] = (transposed) {}", index + 1, index + 1, index + 1, new_bias.transpose());

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
                ActivationFunctionType::HyperbolicTangent => println!("(1 - tanh(z^[{}])^2)", rev_layer_index),
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
        for (index, delta) in deltas.iter().rev().enumerate() {
            let rev_index = deltas.len() - index;

            print!("delta^[{}] = ", rev_index);
            if index == 0 {
                print!("dE / dx^[{}] o dx^[{}] / dz^[{}] = ", rev_index, rev_index, rev_index);
                println!("(transposed) {}", delta.transpose());
                
            } else {
                print!("( dz^[{}] / dx^[{}] ) ^T . delta^[{}] o dx^[{}] / dz^[{}] = ", rev_index + 1, rev_index, rev_index + 1, rev_index, rev_index);
                println!("(transposed) {}", delta.transpose());
            }
        }
    }
}

pub fn build_network(points: Vec<PointInfo>, learning_rate: f64) -> Network {
    let network : Network = Network {
        points: points,
        learning_rate: learning_rate,
        iteration_type: ITERATION,
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
        ActivationFunctionType::HyperbolicTangent => apply_function_to_vector(z, hyperbolic_tangent),
    } 
}

fn square(value: Element) -> Element {
    return value.powi(2);
}

fn sigmoid(value: Element) -> Element {
    let denominator = 1.0 + (- value).exp();
    return 1.0 / denominator;
}

fn hyperbolic_tangent(value: Element) -> Element {
    let denominator = 1.0 + (-2.0 * value).exp();
    return 2.0 / denominator - 1.0;
}

fn apply_function_to_vector(vector: &Vector, function: fn(Element) -> Element) -> Vector {

    let mut values : Vec<Element> = Vec::new();
    for &value in vector.iter() { values.push(function(value)) }

    let new_vector : Vector = Vector::from_vec(values);
    return new_vector;
}