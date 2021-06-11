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
    let likelihoods = load_likelihoods(number_clusters, dimension);
    println!();

    let mut problem : lib::EMBayesian = lib::build_em_bayesian(dimension, x, priors, likelihoods);
    for _ in 0..iterations {
        problem.do_iteration();
        println!();
    }
    problem.compute_final_data_probability();
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

pub fn load_likelihoods(number_clusters: i32, dimension: usize) -> Vec<Vec<lib::Element>> {
    
    let mut likelihoods : Vec<Vec<lib::Element>> = Vec::new();
    for cluster_number in 0..number_clusters {
        let mut sub_likelihoods : Vec<lib::Element> = Vec::new();

        for point_dimension in 0..dimension {
            let line = format!("What is P( x_{} = 1 | C = {} )?", point_dimension + 1, cluster_number + 1);
            let input = ask_for_input(&line);
    
            // Process input line
            let value = input.trim().parse().unwrap();
            sub_likelihoods.push(value);
        }

        likelihoods.push(sub_likelihoods);
    }

    return likelihoods;
}