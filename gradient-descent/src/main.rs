use std::io::{self, Write};

mod lib;

fn main() {
    
    let initial_weights = load_weights();
    let learning_rate = load_learning_rate();
    println!();

    // Get number of points
    let mut line = "What is the number of points you want to give?";
    let buffer = ask_for_input(line);
    let number_points : usize = buffer.trim().parse().unwrap();
    println!();

    let points = load_points(number_points);
    println!();
    let targets = load_targets();
    println!();

    // Get number of iterations
    line = "What is the number of iterations you want to run?";
    let buffer = ask_for_input(line);
    let iterations : usize = buffer.trim().parse().unwrap();
    println!();

    let mut problem : lib::GradientDescent = lib::build_gradient_descent(initial_weights, learning_rate, points, targets);
    problem.run_n_iterations(iterations);
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

pub fn load_weights() -> lib::Vector {
    let line = "What are the initial weights?";
    let input = ask_for_input(line);

    // Process input line
    let values_vec = input.trim().split_whitespace()
        .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

    let value = lib::Vector::from_vec(values_vec);
    return value;
}

pub fn load_learning_rate() -> f64 {
    let line = "What is the learning rate?";
    let input = ask_for_input(line);

    // Process input line
    let value = input.trim().parse().unwrap();

    return value;
}

pub fn load_points(number_points: usize) -> Vec<lib::Vector> {

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

pub fn load_targets() -> lib::Vector {
    let line = "What are the targets?";
    let input = ask_for_input(line);

    // Process input line
    let values_vec = input.trim().split_whitespace()
        .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

    let value = lib::Vector::from_vec(values_vec);
    return value;
}