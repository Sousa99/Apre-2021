use std::io::{self, Write};

mod lib;

fn main() {

    // Get number of points
    let mut line = "What is the number of points you want to give?";
    let buffer = ask_for_input(line);
    let number_points : usize = buffer.trim().parse().unwrap();
    println!();

    let points = load_points(number_points);
    println!();

    println!("==============================================================================");
    let mut problem : lib::SVM = lib::build_svm(points);
    problem.do_computations();
    println!("==============================================================================");

    // Get number of points
    line = "What is the number of points you want to test in the SVM?";
    let buffer = ask_for_input(line);
    let number_points_to_test : usize = buffer.trim().parse().unwrap();
    println!();

    let test_points = load_test_points(number_points_to_test);
    println!();

    for point in test_points.iter() { problem.run_point(point); }
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

pub fn load_points(number_points: usize) -> Vec<lib::PointInfo> {

    let mut points : Vec<lib::PointInfo> = Vec::new();    
    for _ in 0..number_points {

        let line = "What is the next point?";
        let input = ask_for_input(line);

        // Process input line
        let values_vec_point = input.trim().split_whitespace()
            .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

        let value_point = lib::Vector::from_vec(values_vec_point);

        // ===============================================================
        
        let line = "What is its target?";
        let input = ask_for_input(line);
        
        // Process input line
        let value_target = input.trim().parse().unwrap();
        
        // ===============================================================

        let line = "What is its alpha?";
        let input = ask_for_input(line);
        
        // Process input line
        let alpha : lib::Element = input.trim().parse().unwrap();
        
        let point_info : lib::PointInfo = lib::build_point_info(value_point, value_target, alpha);
        points.push(point_info);
    }

    return points;
}


pub fn load_test_points(number_points: usize) -> Vec<lib::Vector> {

    let mut points : Vec<lib::Vector> = Vec::new();    
    for _ in 0..number_points {

        let line = "What is the next point?";
        let input = ask_for_input(line);

        // Process input line
        let values_vec_point = input.trim().split_whitespace()
            .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

        let value_point = lib::Vector::from_vec(values_vec_point);
        points.push(value_point);
    }

    return points;
}