use nalgebra::{DVector};

pub type Element = f64;
pub type Vector = DVector<Element>;

#[allow(dead_code)]
#[derive(Copy, Clone)]
enum IterationType {
    Predefined,
    Stochastic,
}

#[allow(dead_code)]
#[derive(Copy, Clone)]
enum ErrorFunctionType {
    HalfSumSquaredError,
    CrossEntropy,
}

#[allow(dead_code)]
#[derive(Copy, Clone)]
enum ActivationFunctionType {
    Sigmoid,
    Sigmoid2,
    SquaredExponential
}

// ============================ START PREDEFINED PARAMETERS ============================

const ITERATION : IterationType = IterationType::Predefined;
const ERROR : ErrorFunctionType = ErrorFunctionType::HalfSumSquaredError;
const ACTIVATION : ActivationFunctionType = ActivationFunctionType::Sigmoid2;

// ============================ END PREDEFINED PARAMETERS ============================

pub struct GradientDescent {
    initial_weights: Vector,
    learning_rate: f64,
    points: Vec<Vector>,
    targets: Vector,
    // Iteration Process
    iteration_type: IterationType,
    error_function: ErrorFunctionType,
    activation_function: ActivationFunctionType,
    // Iteration information
    weights: Vec<Vector>,
}

impl GradientDescent {

    pub fn run_n_iterations(&mut self, number_iterations: usize) {
        for _ in 0..number_iterations {
            println!("Epoch {}", self.weights.len());
            println!();
            println!("=================================================================");

            self.run_iteration();
            println!("  Weight (final) = {}", self.weights.last().unwrap().transpose());
            println!();
        }
    }

    fn run_iteration(&mut self) {

        let dimension : usize = self.initial_weights.shape().0;
        let last_weight : Vector = self.weights.last().unwrap().clone();
        let new_weight : Vector = match self.iteration_type {
            IterationType::Predefined => {

                let mut variation : Vector = Vector::from_element(dimension, 0.0);
                for (point_number, (point, &target)) in self.points.iter().zip(self.targets.iter()).enumerate() {
                    let sub_point : Vector = calculate_sub_point(&last_weight, point, target, self.error_function, self.activation_function);
                    variation = variation + sub_point.clone();
                    println!("      Point {} = {}", point_number + 1, sub_point.transpose());
                };

                last_weight - self.learning_rate * variation
            },

            IterationType::Stochastic => {

                let mut tmp_weight = last_weight.clone();
                for (point_number, (point, &target)) in self.points.iter().zip(self.targets.iter()).enumerate() {

                    let sub_point : Vector = calculate_sub_point(&last_weight, point, target, self.error_function, self.activation_function);
                    tmp_weight = tmp_weight - self.learning_rate * sub_point.clone();
                    
                    println!("      Point {} = {}", point_number + 1, sub_point.transpose());
                    println!("      Point Weight = {}", tmp_weight.transpose());
                };

                tmp_weight
            }
        };

        self.weights.push(new_weight.clone());
    }
}

pub fn build_gradient_descent(initial_weights: Vector, learning_rate: f64, points: Vec<Vector>, targets: Vector) -> GradientDescent {
    let gradient_descent: GradientDescent = GradientDescent {
        initial_weights: initial_weights.clone(),
        learning_rate: learning_rate,
        points: points,
        targets: targets,
        // Iteration Process
        iteration_type: ITERATION,
        error_function: ERROR,
        activation_function: ACTIVATION,
        // Iteration information
        weights: vec![initial_weights]
    };

    return gradient_descent;
}

fn calculate_sub_point(weight: &Vector, point: &Vector, target: f64, error_function: ErrorFunctionType, activation_function: ActivationFunctionType) -> Vector {

    // Auxiliary values
    let weight_point : f64 = (weight.transpose() * point)[(0, 0)];

    let output = match activation_function {
        ActivationFunctionType::Sigmoid => sigmoid(weight_point),
        ActivationFunctionType::Sigmoid2 => sigmoid(2.0 * weight_point),
        ActivationFunctionType::SquaredExponential => exponential(weight_point.powi(2)),
    };

    // Caclulate sub point error
    let sub_point : Vector = match (error_function, activation_function) {
        (ErrorFunctionType::HalfSumSquaredError, ActivationFunctionType::Sigmoid2) => - 2.0 * point * (target - output) * output * (1.0 - output),
        (ErrorFunctionType::CrossEntropy, ActivationFunctionType::Sigmoid) => - point * (target - output),
        (ErrorFunctionType::HalfSumSquaredError, ActivationFunctionType::SquaredExponential) => - 2.0 * (target - output) * output * weight_point * point,
        _ => panic!("Combination not defined"),
    };

    return sub_point;
}

// ==================== AUXILIARY FUNCTIONS ====================

fn sigmoid(value: f64) -> f64 {
    let denominator = 1.0 + (- value).exp();
    return 1.0 / denominator;
}

fn exponential(value: f64) -> f64 {
    return value.exp();
}