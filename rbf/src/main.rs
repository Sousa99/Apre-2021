use nalgebra::{DMatrix};
use std::io::{self, Write};

mod lib;

fn main() {

    // Number of iterations
    let mut line : &str = "How many iterations do you want computed?";
    let buffer = ask_for_input(line);
    let iterations : usize = buffer.trim().parse().unwrap();

    // Get dimension
    line = "What dimension are we dealing with?";
    let buffer = ask_for_input(line);
    let dimension : usize = buffer.trim().parse().unwrap();
    
    // Get number of points
    line = "What is the number of points given?";
    let buffer = ask_for_input(line);
    let number_points : usize = buffer.trim().parse().unwrap();
    
    // Get number of clusters
    line = "What is the number of clusters that we are dealing with?";
    let buffer = ask_for_input(line);
    let number_clusters : usize = buffer.trim().parse().unwrap();

    
    println!();

    let x = load_x(number_points as i32, dimension);
    println!();
    let targets = load_targets(number_points as i32);
    println!();
    let initial_weights = load_weights(number_clusters as i32);
    println!();
    let learning_rate = load_learning_rate();
    println!();
    let cluster_centers = load_cluster_centers(number_clusters as i32, dimension);
    println!();
    let cluster_sigmas = load_cluster_sigmas(number_clusters as i32);
    println!();

    let mut problem : lib::RBF = lib::build_rbf(number_points, number_clusters, x, targets, initial_weights, learning_rate, cluster_centers, cluster_sigmas);
    problem.do_n_iterations(iterations);

    // Get number of points to test
    line = "What is the number of points to test the network?";
    let buffer = ask_for_input(line);
    let number_points_to_test : usize = buffer.trim().parse().unwrap();

    // Load Points to test
    let x_test = load_x(number_points_to_test as i32, dimension);
    println!();
    
    println!("=============== Testing Points ===============");
    println!();
    // Run points through network
    problem.run_points(x_test);
}

pub fn ask_for_input(line: &str) -> String {
    println!("{}", line);
    print!("> ");
    
    let stdin : std::io::Stdin = io::stdin();
    let mut buffer : String = String::new();
    let _ = io::stdout().flush();
    stdin.read_line(&mut buffer).unwrap();

    return buffer;
}

pub fn load_x(number_points: i32, dimension: usize) -> Vec<lib::Matrix> {

    let mut points : Vec<lib::Matrix> = Vec::new();    
    for _ in 0..number_points {

        let line = "What is the next point?";
        let input = ask_for_input(line);

        // Process input line
        let values_vec = input.trim().split_whitespace()
            .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

        let value = DMatrix::from_vec(dimension, 1, values_vec);
        points.push(value);
    }

    return points;
}

pub fn load_targets(number_points: i32) -> lib::Matrix {

    let line = "What are the targets for the points (all together)?";
    let input = ask_for_input(line);

    // Process input line
    let values_vec = input.trim().split_whitespace()
        .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

    let value = DMatrix::from_vec(number_points as usize, 1, values_vec);

    return value;
}

pub fn load_weights(number_clusters: i32) -> lib::Matrix {

    let line = "What is the initial weight matrix?";
    let input = ask_for_input(line);

    // Process input line
    let values_vec = input.trim().split_whitespace()
        .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

    let value = DMatrix::from_vec((number_clusters + 1) as usize, 1, values_vec);

    return value;
}

pub fn load_learning_rate() -> lib::Element {

    let line = "What is the learning rate?";
    let input = ask_for_input(line);

    // Process input line
    let value = input.trim().parse().unwrap();

    return value;
}

pub fn load_cluster_centers(number_clusters: i32, dimension: usize) -> Vec<lib::Matrix> {

    let mut points : Vec<lib::Matrix> = Vec::new();    
    for _ in 0..number_clusters {

        let line = "What is the next cluster?";
        let input = ask_for_input(line);

        // Process input line
        let values_vec = input.trim().split_whitespace()
            .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

        let value = DMatrix::from_vec(dimension, 1, values_vec);
        points.push(value);
    }

    return points;
}

pub fn load_cluster_sigmas(number_clusters: i32) -> Vec<lib::Element> {

    let mut points : Vec<lib::Element> = Vec::new();    
    for _ in 0..number_clusters {

        let line = "What is the sigma for the next cluster?";
        let input = ask_for_input(line);

        // Process input line
        let value = input.trim().parse().unwrap();
        points.push(value);
    }

    return points;
}