use std::io::{self, Write};
mod lib;

fn main() {

    // Number of iterations
    let mut line : &str = "How many iterations do you want computed?";
    let buffer = ask_for_input(line);
    let iterations : i32 = buffer.trim().parse().unwrap();

    // Get number of points
    line = "What is the number of points given?";
    let buffer = ask_for_input(line);
    let number_points : i32 = buffer.trim().parse().unwrap();

    println!();

    let points = load_points(number_points);
    println!();
    let targets = load_targets(number_points);
    println!();

    let weights = load_initial_weigths();
    println!();
    let learning_rate = load_learning_rate();
    println!();

    let mut problem : lib::Perceptron = lib::build_perceptron(weights, learning_rate);
    if iterations == -1 { problem.run_until_convergence(&points, &targets); }
    else {
        for _ in 0..iterations {
            problem.run_iteration(&points, &targets);
        }
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

pub fn load_points(number_points: i32) -> Vec<lib::Point> {
    let mut points : Vec<lib::Point> = Vec::new();    
    for point_number in 0..number_points {

        let line = format!("What is the point {} (with bias in it)?", point_number);
        let input = ask_for_input(&line);

        // Process input line
        let value = input.trim().split_whitespace()
            .map(|value| value.parse().unwrap()).collect::<lib::Point>();
        
        points.push(value);
    }

    return points;
}

pub fn load_targets(number_points: i32) -> Vec<f64> {
    let mut targets : Vec<f64> = Vec::new();
    for point_number in 0..number_points {

        let line = format!("What is the target for the point {}?", point_number);
        let input = ask_for_input(&line);

        // Process input line
        let value = input.trim().parse().unwrap();
        targets.push(value);
    }

    return targets;
}

pub fn load_initial_weigths() -> lib::Weights  {

    let line = format!("What are the initial weights?");
    let input = ask_for_input(&line);

    // Process input line
    let value = input.trim().split_whitespace()
        .map(|value| value.parse().unwrap()).collect::<lib::Weights>();

    return value;
}

pub fn load_learning_rate() -> f64 {

    let line = format!("What is the learning rate?");
    let input = ask_for_input(&line);

    // Process input line
    let value = input.trim().parse().unwrap();
    return value;
}