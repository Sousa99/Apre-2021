use std::io::{self, Write};

mod lib;

fn main() {
    // Get number of points
    let mut line = "What is the number of points you want to calculate?";
    let buffer = ask_for_input(line);
    let number_points : usize = buffer.trim().parse().unwrap();

    println!();

    let points = load_points(number_points);
    println!();
    let targets = load_targets();
    println!();

    // Get number of points
    line = "What is the lamda to be used?";
    let buffer = ask_for_input(line);
    let lambda : f64 = buffer.trim().parse().unwrap();

    let problem : lib::ClosedForm = lib::build_closed_form(points, targets, lambda);
    let weights = problem.compute_weights();
    println!("Final weight = {}", weights);
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

pub fn load_points(number_points: usize) -> lib::Matrix {

    let mut values : Vec<lib::Element> = Vec::new();
    for _ in 0..number_points {

        let line = "What is the next point (include bias)?";
        let input = ask_for_input(line);

        // Process input line
        let mut values_vec = input.trim().split_whitespace()
            .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

        values.append(&mut values_vec);
    }

    let dimension = values.len() / number_points;
    let points = lib::Matrix::from_vec(dimension, number_points, values);
    let points_corrected = points.transpose();
    println!("{}", points_corrected);

    return points_corrected;
}

pub fn load_targets() -> lib::Vector {
    let line = "What are the targets?";
    let input = ask_for_input(line);

    // Process input line
    let values_vec = input.trim().split_whitespace()
        .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

    let value = lib::Vector::from_vec(values_vec);
    return value;
}