pub type Weights = Vec<f64>;
pub type Point = Vec<f64>;
pub type Output = f64;

pub struct Perceptron {
    step: i64,
    weights: Weights,
    learning_rate: f64,
}

impl Perceptron {

    fn run_point_update(&mut self, point_identifier: String, point: &Point, target: Output) -> bool {

        let class_output = calculate_output(&self.weights, point);
        println!("  Point {} ({:?}): target = {}, output = {}", point_identifier, point, target, class_output);
        if class_output == target { return false; }

        // Did not get it correctly
        let new_weights = update_weights(&self.weights, self.learning_rate, point, target, class_output);
        println!("        UPDATE: {:?} -> {:?}", self.weights, new_weights);

        self.weights = new_weights;
        return true;
    }

    pub fn run_iteration(&mut self, points: &Vec<Point>, targets: &Vec<f64>) -> bool {
        // Update Epoch
        self.step = self.step + 1;
        let mut converged = true;

        println!("Epoch {}:", self.step);
        for (index, (point, &target)) in points.iter().zip(targets.iter()).enumerate() {
            let point_identifier = index.to_string();
            let updated = self.run_point_update(point_identifier, point, target);

            converged = converged && (!updated);
        }

        return converged;
    }

    pub fn run_until_convergence(&mut self, points: &Vec<Point>, targets: &Vec<f64>) {

        let mut converged = false;
        while !converged { converged = self.run_iteration(points, targets); }
    }
}

pub fn build_perceptron(weights: Weights, learning_rate: f64) -> Perceptron {
    let perceptron : Perceptron = Perceptron {
        step: 0,
        weights: weights,
        learning_rate: learning_rate,
    };

    return perceptron;
}

// ===========================================================================

fn calculate_output(weights: &Weights, point: &Point) -> f64 {
    let mut output_value : f64 = 0.0;
    for (weight, point_value) in weights.iter().zip(point.iter()) {
        output_value = output_value + weight * point_value;
    }

    if output_value >= 0.0 { return 1.0 }
    else { return -1.0 }
}

fn update_weights(weights: &Weights, learning_rate: f64, point: &Point, target:f64, output: f64) -> Weights {
    let mut new_weigths : Weights = weights.clone();

    // Update calculation
    let factor = learning_rate * (target - output);
    for (index, point_value) in point.iter().enumerate() {
        new_weigths[index] = new_weigths[index] + factor * point_value;
    }

    return new_weigths;
}