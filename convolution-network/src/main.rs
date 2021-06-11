use std::io::{self, Write};

mod lib;

fn main() {

    let dimension = load_dimension();
    println!();
    let input = load_input(dimension);
    println!();

    let problem : lib::ConvolutionNetwork = lib::build_convolution_network();
    problem.run_input(input);
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

pub fn load_dimension() -> (usize, usize, usize) {

    let line = "What is input dimension?";
    let input = ask_for_input(line);

    // Process input line
    let values_vec = input.trim().split_whitespace()
        .map(|value| value.parse().unwrap()).collect::<Vec<usize>>();

    return (values_vec[0], values_vec[1], values_vec[2]);
}

pub fn load_input(dimension: (usize, usize, usize)) -> Vec<lib::Matrix> {

    let mut inputs: Vec<lib::Matrix> = Vec::new();
    for _ in 0..dimension.2 {
        let line = "What is the input matrix (channel by channel) ?";
        let input = ask_for_input(line);
    
        // Process input line
        let values_vec = input.trim().split_whitespace()
            .map(|value| value.parse().unwrap()).collect::<Vec<lib::Element>>();

        let input = lib::Matrix::from_vec(dimension.0, dimension.1, values_vec);
        inputs.push(input.transpose());
    }

    return inputs;
}