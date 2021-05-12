use nalgebra::{DMatrix};

pub type Element = f64;
pub type Matrix = DMatrix<Element>;

pub struct RBF {
    number_points: usize,
    number_clusters: usize,
    x: Vec<Matrix>,
    targets: Matrix,
    initial_weights: Matrix,
    learning_rate: Element,
    cluster_centers: Vec<Matrix>,
    cluster_sigmas: Vec<Element>,
}

impl RBF {

    pub fn do_n_iterations(&self, iterations: usize) {
        // Compute Radial Layer Matrix
        let radial_matrix = self.compute_radial_matrix();
        println!("Radial Matrix = Φ = {}", radial_matrix);

        // Split Radial Matrix by Row
        let radial_matrix_split = split_matrix_by_row(radial_matrix, self.number_points, self.number_clusters + 1);

        // Store weights
        let mut weights = self.initial_weights.clone();
        println!("Weights at 0: {}", weights);

        // Do N iterations to the weights
        // TODO: Implement more than cross-entropy with sigmoid function
        for iteration in 0..iterations {
            weights = cross_entropy_sigmoid_update(weights, radial_matrix_split.clone(), self.targets.clone(), self.learning_rate);
            println!("Weights at {}: {}", iteration + 1, weights);
        }
    }

    pub fn compute_radial_matrix(&self) -> Matrix {

        let mut radial_values : Vec<Element> = Vec::new();

        // Add bias column values
        for _ in 0..self.number_points { radial_values.push(1.0) }
        // Add values column by column
        for (cluster_center, &cluster_sigma) in self.cluster_centers.iter().zip(self.cluster_sigmas.iter()) {
            for point in self.x.iter() {
                let new_radial_value = radial_function(point.clone(), cluster_center.clone(), cluster_sigma);
                radial_values.push(new_radial_value);
            }
        }

        let radial_matrix = DMatrix::from_vec(self.number_points, self.number_clusters + 1, radial_values);
        return radial_matrix;
    }
}

pub fn build_rbf(number_points: usize, number_clusters: usize, x: Vec<Matrix>, targets: Matrix, initial_weights: Matrix, learning_rate: Element, cluster_centers: Vec<Matrix>, cluster_sigmas: Vec<Element>) -> RBF {
    let new_rbf : RBF = RBF {
        number_points: number_points,
        number_clusters: number_clusters,
        x: x,
        targets: targets,
        initial_weights: initial_weights,
        learning_rate: learning_rate,
        cluster_centers: cluster_centers,
        cluster_sigmas: cluster_sigmas,
    };

    return new_rbf;
}

pub fn radial_function(x: Matrix, cluster_center: Matrix, cluster_sigma: Element) -> Element {
    
    // Compute distance
    let mut distance = 0.0;
    for (&x_value, &cluster_center_value) in x.iter().zip(cluster_center.iter()) {
        distance = distance + (x_value - cluster_center_value).powi(2);
    }

    // Compute radial value
    let denominator = 2.0 * cluster_sigma.powi(2);
    let radial_value = (- distance / denominator).exp();

    return radial_value;
}

pub fn cross_entropy_sigmoid_update(previous_weights: Matrix, radial_matrix: Vec<Matrix>, targets: Matrix, learning_rate: Element) -> Matrix {
    
    let mut new_weights = previous_weights.clone();
    for (point, &target) in radial_matrix.iter().zip(targets.iter()) {

        let output = sigmoid(previous_weights.clone(), point.clone());
        let current_computation = learning_rate * point * (target - output);
        new_weights = new_weights + current_computation;
    }

    return new_weights;
}

pub fn sigmoid(weights: Matrix, point: Matrix) -> Element {

    // Multiply weights by point
    let mut value = 0.0;
    for (&weight, &point_value) in weights.iter().zip(point.iter()) {
        value = value + weight * point_value;
    }

    // Aplly sigmoid function
    value = 1.0 / (1.0 + (-value).exp());

    return value;
}

fn split_matrix_by_row(matrix: Matrix, number_rows: usize, number_columns: usize) -> Vec<Matrix> {

    let mut final_result : Vec<Matrix> = Vec::new();

    let mut temp_values : Vec<Vec<Element>> = Vec::new();
    for _ in 0..number_rows { temp_values.push(Vec::new()); }

    for (index, &value) in matrix.iter().enumerate() {
        let row = index % number_rows;
        temp_values[row].push(value);
    }

    for line in temp_values {
        let sub_matrix = DMatrix::from_vec(number_columns, 1, line);
        final_result.push(sub_matrix);
    }

    return final_result;
}