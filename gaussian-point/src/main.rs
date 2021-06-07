use std::io::{self, Write};

mod lib;

fn main() {

    let mean = load_mean();
    println!();
    let covariance = load_covariance();
    println!();

    // Get number of points
    let line = "What is the number of points you want to calculate?";
    let buffer = ask_for_input(line);
    let number_points : i32 = buffer.trim().parse().unwrap();

    println!();

    let points = load_points(number_points);
    println!();

    let problem : lib::Gaussian = lib::build_gaussian(mean, covariance);
    for point in points { problem.run_point(point); }
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

pub fn load_mean() -> lib::Vector {
    let line = "What is the mean?";
    let input = ask_for_input(line);

    // Process input line
    let values_vec = input.trim().split_whitespace()
        .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

    let value = lib::Vector::from_vec(values_vec);
    return value;
}

pub fn load_covariance() -> lib::Matrix {

    let line = format!("What is the covariance?");
    let input = ask_for_input(&line);

    // Process input line
    let values_vec = input.trim().split_whitespace()
        .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

    let dimension = (values_vec.len() as f64).sqrt() as usize;
    let value = lib::Matrix::from_vec(dimension, dimension, values_vec);
    return value;
}

pub fn load_points(number_points: i32) -> Vec<lib::Vector> {

    let mut points : Vec<lib::Vector> = Vec::new();    
    for _ in 0..number_points {

        let line = "What is the next point?";
        let input = ask_for_input(line);

        // Process input line
        let values_vec = input.trim().split_whitespace()
            .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

        let value = lib::Vector::from_vec(values_vec);
        points.push(value);
    }

    return points;
}