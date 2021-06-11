use nalgebra::{DMatrix};
use itertools::izip;
use std::collections::HashMap;

pub type Element = f64;
pub type Matrix = DMatrix<Element>;
// ======================== AUXILIARY TYPES ========================
type PosteriorsReturn = HashMap<(usize, usize), Element>;

#[derive(Clone)]
pub struct EMBayesianParameter {
    prior: f64,
    likelihoods: Vec<Element>,
}

pub struct EMBayesian {
    step: i32,
    dimension: usize,
    x: Vec<Matrix>,
    parameters: Vec<EMBayesianParameter>,
}

impl EMBayesian {

    pub fn do_iteration(&mut self) {
        self.step = self.step + 1;
        println!("==================================== STEP {} ====================================", self.step);
        
        let posteriors = self.do_e_step();
        self.compute_data_probability(false);
        let new_params = self.do_m_step(posteriors);
        self.parameters = new_params;
    }

    pub fn compute_final_data_probability(&self) {
        println!("==================================== FINAL DATA PROBABILITY COMPUTATION ====================================");
        self.compute_data_probability(true);
    }

    fn do_e_step(&mut self) -> PosteriorsReturn {
        println!("E Step:");

        let mut posteriors : PosteriorsReturn = PosteriorsReturn::new();
        for (point_index, point) in self.x.iter().enumerate() {
            println!();
            println!("\t- For x({}):", point_index + 1);

            let mut joint_probabilities : Vec<Element> = Vec::new();
            let mut sum_joint_probabilities : Element = 0.0;
            
            for (cluster_index, parameter) in self.parameters.iter().enumerate() {
                println!("\t\t- For Cluster = {}:", cluster_index + 1);
                
                // Do Calculations
                let prior = parameter.prior;
                let likelihood = compute_likelihood(point.clone(), parameter.likelihoods.clone());
                let joint_probability = compute_joint_probability(prior, likelihood);
                
                // Print Results
                println!("\t\t\t- Prior: p(C = {}) = {}:", cluster_index + 1, prior);
                println!("\t\t\t- Likelihood: p(x^({}) | C = {}) = {}:", point_index + 1, cluster_index + 1, likelihood);
                println!("\t\t\t- Joint Probability: p(C = {}, x^({})) = {}:", cluster_index + 1, point_index + 1, joint_probability);

                // Update stored values
                joint_probabilities.push(joint_probability);
                sum_joint_probabilities = sum_joint_probabilities + joint_probability;
            }

            println!();

            println!("\t\t- Updating normalized posteriors:");
            for (cluster_index, joint_probability) in joint_probabilities.iter().enumerate() {
                
                // Do Calculations
                let normalized_posterior : Element = joint_probability / sum_joint_probabilities;
                posteriors.insert((cluster_index + 1, point_index + 1), normalized_posterior);
                
                // Print Results
                println!("\t\t\t- C = {}: p(C = {} | x^({}))= {}:", cluster_index + 1, cluster_index + 1, point_index + 1, normalized_posterior);
            }
        }
            
        return posteriors;
    }
    
    fn do_m_step(&mut self, posteriors: PosteriorsReturn ) -> Vec<EMBayesianParameter> {
        println!("M Step:");

        let mut new_params : Vec<EMBayesianParameter> = Vec::new();

        let sum_posteriors_all : Element = sum_of_all_posteriors(&posteriors);
        for cluster_index in 0..self.parameters.len() {
            println!();
            println!("\t- For C = {}:", cluster_index + 1);

            // ====== Compute Sum of Posteriors ======
            let mut sum_posteriors_cluster : Element = 0.0;
            for point_index in 0..self.x.len() {
                let posterior : Element = *posteriors.get(&(cluster_index + 1, point_index + 1)).unwrap();
                sum_posteriors_cluster = sum_posteriors_cluster + posterior;
            }

            // ====== Compute New Likelihoods ======
            let mut likelihoods : Vec<Element> = Vec::new();
            for dimension in 0..self.dimension {
                let mut likelihood : Element = 0.0;
                for point_index in 0..self.x.len() {
                    let posterior : Element = *posteriors.get(&(cluster_index + 1, point_index + 1)).unwrap();
                    let dimension_value : Element = self.x[point_index][dimension];

                    if dimension_value == 1.0 { likelihood = likelihood + posterior }
                }

                likelihoods.push(likelihood / sum_posteriors_cluster);
            }

            // ====== Compute New Prior ======
            let new_prior : Element = sum_posteriors_cluster / sum_posteriors_all;

            // ====== Print new Parameters ======
            println!();
            for (dimension, likelihood) in likelihoods.iter().enumerate() {
                println!("\t\tp( x_{} = 1 | C = {} ) = {}", dimension + 1, cluster_index + 1, likelihood);
            }
            println!();
            println!("\t\tPrior: p(C = {}) = {}", cluster_index + 1, new_prior);

            // Create and save back new Params
            let new_param : EMBayesianParameter = EMBayesianParameter {
                prior: new_prior,
                likelihoods: likelihoods,
            };

            new_params.push(new_param);
        }

        return new_params;
    }

    fn compute_data_probability(&self, print: bool) {

        let mut probability : Element = 1.0;
        for (point_index, point) in self.x.iter().enumerate() {

            if print {
                println!();
                println!("- For x({}):", point_index + 1);
            }

            let mut sum_joint_probabilities : Element = 0.0;
            
            for (cluster_index, parameter) in self.parameters.iter().enumerate() {
                if print { println!("\t- For Cluster = {}:", cluster_index + 1) }
                
                // Do Calculations
                let prior = parameter.prior;
                let likelihood = compute_likelihood(point.clone(), parameter.likelihoods.clone());
                let joint_probability = compute_joint_probability(prior, likelihood);
                
                // Print Results
                if print {
                    println!("\t\t- Prior: p(C = {}) = {}:", cluster_index + 1, prior);
                    println!("\t\t- Likelihood: p(x^({}) | C = {}) = {}:", point_index + 1, cluster_index + 1, likelihood);
                    println!("\t\t- Joint Probability: p(C = {}, x^({})) = {}:", cluster_index + 1, point_index + 1, joint_probability);
                }

                // Update stored values
                sum_joint_probabilities = sum_joint_probabilities + joint_probability;
            }

            probability = probability * sum_joint_probabilities;
        }

        println!();
        println!("Data Probability = {}", probability);
        println!();
    }
}

pub fn build_em_bayesian(dimension: usize, x: Vec<Matrix>, priors: Vec<f64>, likelihoods: Vec<Vec<Element>>) -> EMBayesian {

    let mut parameters : Vec<EMBayesianParameter> = Vec::new();
    for (prior, sub_likelihoods) in izip!(priors, likelihoods) {
        let new_parameter : EMBayesianParameter = EMBayesianParameter {
            prior: prior,
            likelihoods: sub_likelihoods,
        };

        parameters.push(new_parameter);
    }

    let new_em : EMBayesian = EMBayesian {
        step: 0,
        dimension: dimension,
        x: x,
        parameters: parameters,
    };

    return new_em;
}

// ============================= UTIL FUNCTIONS =============================
fn compute_likelihood(x: Matrix, likelihoods: Vec<Element>) -> Element {

    let mut result: Element = 1.0;
    for (&point_value, &likelihood) in x.iter().zip(likelihoods.iter()) {
        if point_value == 1.0 { result = result * likelihood }
        else { result = result * ( 1.0 - likelihood ) }
    }

    return result;
}

fn compute_joint_probability(prior: Element, likelihood: Element) -> Element {
    return prior * likelihood;
}

fn sum_of_all_posteriors(posteriors: &PosteriorsReturn) -> Element {
    let mut sum : Element = 0.0;
    for (_, element) in posteriors {
        sum = sum + *element;
    }

    return sum;
}