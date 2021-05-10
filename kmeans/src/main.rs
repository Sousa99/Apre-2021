use nalgebra::{DVector};
use std::io::{self, Write};

mod lib;

fn main() {

    // Number of iterations
    let mut line : &str = "How many iterations do you want computed?";
    let buffer = ask_for_input(line);
    let iterations : usize = buffer.trim().parse().unwrap();

    // Export Image
    line = "Do you wish to export as images the steps?";
    let buffer = ask_for_input(line);
    let export_image : bool = buffer.trim().parse().unwrap();
    
    // Get number of points
    line = "What is the number of points given?";
    let buffer = ask_for_input(line);
    let number_points : i32 = buffer.trim().parse().unwrap();
    
    // Get number of clusters
    line = "What is the number of clusters that we are dealing with?";
    let buffer = ask_for_input(line);
    let number_clusters : i32 = buffer.trim().parse().unwrap();

    
    println!();

    let x = load_x(number_points);
    println!();
    let cluster_centers = load_cluster_centers(number_clusters);
    println!();

    let mut problem : lib::KMeans = lib::build_kmeans(export_image, x, cluster_centers);
    problem.do_loop_iterations(iterations);
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

pub fn load_x(number_points: i32) -> Vec<lib::Vector> {

    let mut points : Vec<lib::Vector> = Vec::new();    
    for _ in 0..number_points {

        let line = "What is the next point?";
        let input = ask_for_input(line);

        // Process input line
        let values_vec = input.trim().split_whitespace()
            .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

        let value = DVector::from_vec(values_vec);
        points.push(value);
    }

    return points;
}

pub fn load_cluster_centers(number_clusters: i32) -> Vec<lib::Vector> {
    let mut points : Vec<lib::Vector> = Vec::new();    
    for _ in 0..number_clusters {

        let line = "What is the cluster center?";
        let input = ask_for_input(line);

        // Process input line
        let values_vec = input.trim().split_whitespace()
            .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

        let value = DVector::from_vec(values_vec);
        points.push(value);
    }

    return points;
}