use std::io::{self, Write};

mod lib;

fn main() {
    // Get number of points
    let line = "What is the number of points given?";
    let buffer = ask_for_input(line);
    let number_points : i32 = buffer.trim().parse().unwrap();

    println!();

    let points = load_x(number_points);
    println!();

    let mut problem : lib::Gaussian = lib::build_gaussian(points);
    problem.compute_parameters();
    problem.print_parameters();
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

        let value = lib::Vector::from_vec(values_vec);
        points.push(value);
    }

    return points;
}