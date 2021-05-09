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
    
    // Get number of clusters
    line = "What is the number of clusters that we are dealing with?";
    let buffer = ask_for_input(line);
    let number_clusters : i32 = buffer.trim().parse().unwrap();

    // Get number of points
    line = "What is the number of points given?";
    let buffer = ask_for_input(line);
    let number_points : i32 = buffer.trim().parse().unwrap();
    
    println!();

    let x = load_x(number_points, dimension);
    println!();
    let priors = load_priors(number_clusters);
    println!();
    let means = load_means(number_clusters, dimension);
    println!();
    let covariances = load_covariances(number_clusters, dimension);
    println!();

    let mut problem : lib::EMMultivariate = lib::build_em_multivariate(dimension, x, priors, means, covariances);
    for _ in 0..iterations {
        problem.do_iteration();
        println!();
    }
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

pub fn load_priors(number_clusters: i32) -> Vec<lib::Element> {
    let mut priors : Vec<lib::Element> = Vec::new();    
    for cluster_number in 0..number_clusters {

        let line = format!("What is the prior for cluster {}?", cluster_number);
        let input = ask_for_input(&line);

        // Process input line
        let value = input.trim().parse().unwrap();
        priors.push(value);
    }

    return priors;
}

pub fn load_means(number_clusters: i32, dimension: usize) -> Vec<lib::Matrix> {
    
    let mut means : Vec<lib::Matrix> = Vec::new();
    for cluster_number in 0..number_clusters {

        let line = format!("What is the mean for cluster {}?", cluster_number);
        let input = ask_for_input(&line);

        // Process input line
        let values_vec = input.trim().split_whitespace()
            .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

        let value = DMatrix::from_vec(dimension, 1, values_vec);
        means.push(value);
    }

    return means;
}

pub fn load_covariances(number_clusters: i32, dimension: usize) -> Vec<lib::Matrix> {
    
    let mut covariances : Vec<lib::Matrix> = Vec::new();
    for cluster_number in 0..number_clusters {

        let line = format!("What is the covariance for cluster {}?", cluster_number);
        let input = ask_for_input(&line);

        // Process input line
        let values_vec = input.trim().split_whitespace()
            .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

        let value = DMatrix::from_vec(dimension, dimension, values_vec);
        covariances.push(value);
    }

    return covariances;
}
