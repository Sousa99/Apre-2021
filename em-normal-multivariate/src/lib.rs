use nalgebra::{DMatrix};
use itertools::izip;
use std::f64::consts::PI;
use std::collections::HashMap;

pub type Element = f64;
pub type Matrix = DMatrix<Element>;
// ======================== AUXILIARY TYPES ========================
type PosteriorsReturn = HashMap<(usize, usize), Element>;

#[derive(Clone)]
pub struct EMMultivariateParameter {
    prior: f64,
    mean: Matrix,
    covariance: Matrix,
}

pub struct EMMultivariate {
    step: i32,
    dimension: usize,
    x: Vec<Matrix>,
    parameters: Vec<EMMultivariateParameter>,
}

impl EMMultivariate {

    pub fn do_iteration(&mut self) {
        self.step = self.step + 1;
        println!("==================================== STEP {} ====================================", self.step);

        let posteriors = self.do_e_step();
        let new_params = self.do_m_step(posteriors);
        self.parameters = new_params;
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
                let likelihood = compute_likelihood(point.clone(), parameter.mean.clone(), parameter.covariance.clone());
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
    
    fn do_m_step(&mut self, posteriors: PosteriorsReturn ) -> Vec<EMMultivariateParameter> {
        println!("M Step:");

        let mut new_params : Vec<EMMultivariateParameter> = Vec::new();

        let sum_posteriors_all : Element = sum_of_all_posteriors(&posteriors);
        for (cluster_index, old_param) in self.parameters.iter().enumerate() {
            println!();
            println!("\t- For C = {}:", cluster_index + 1);

            // ====== Compute Sum of Posteriors ======
            let mut sum_posteriors_cluster : Element = 0.0;
            for point_index in 0..self.x.len() {
                let posterior : Element = *posteriors.get(&(cluster_index + 1, point_index + 1)).unwrap();
                sum_posteriors_cluster = sum_posteriors_cluster + posterior;
            }

            // ====== Compute New Mean ======
            let mut new_mean : Matrix = create_copy_with_value(old_param.mean.clone(), 0.0);
            for (point_index, point) in self.x.iter().enumerate() {
                let posterior : Element = *posteriors.get(&(cluster_index + 1, point_index + 1)).unwrap();
                new_mean = new_mean + posterior * point;
            }
            new_mean = new_mean / sum_posteriors_cluster;
            
            // ====== Compute New Covariance ======
            let mut new_cov_vector : Vec<Element> = Vec::new();
            for row in 0..self.dimension {
                for column in 0..self.dimension {
                    let mut sub_cov_value : Element = 0.0;

                    for (point_index, point) in self.x.iter().enumerate() {
                        let posterior : Element = *posteriors.get(&(cluster_index + 1, point_index + 1)).unwrap();
                        
                        let temp : Element = posterior * (point[(row, 0)] - new_mean[(row, 0)]) * (point[(column, 0)] - new_mean[(column, 0)]);
                        sub_cov_value = sub_cov_value + temp;
                    }

                    new_cov_vector.push(sub_cov_value);
                }
            }
            let mut new_cov = DMatrix::from_vec(self.dimension, self.dimension, new_cov_vector);
            new_cov = new_cov / sum_posteriors_cluster;

            // ====== Compute New Prior ======
            let new_prior : Element = sum_posteriors_cluster / sum_posteriors_all;

            // ====== Print new Parameters ======
            println!("\t\tμ^{} = {}", cluster_index + 1, new_mean);
            println!();
            for row in 0..self.dimension {
                for column in 0..self.dimension {
                    println!("\t\tΣ^{}_{},{} = {}", cluster_index + 1, row + 1, column + 1, new_cov[(row, column)]);
                }
            }
            println!();
            println!("\t\tPrior: p(C = {}) = {}", cluster_index + 1, new_prior);

            // Create and save back new Params
            let new_param : EMMultivariateParameter = EMMultivariateParameter {
                prior: new_prior,
                mean: new_mean,
                covariance: new_cov,
            };

            new_params.push(new_param);

        }

        return new_params;
    }
}

pub fn build_em_multivariate(dimension: usize, x: Vec<Matrix>, priors: Vec<f64>, means: Vec<Matrix>, covariances: Vec<Matrix>) -> EMMultivariate {

    let mut parameters : Vec<EMMultivariateParameter> = Vec::new();
    for (prior, mean, covariance) in izip!(priors, means, covariances) {
        let new_parameter : EMMultivariateParameter = EMMultivariateParameter {
            prior: prior,
            mean: mean,
            covariance: covariance,
        };

        parameters.push(new_parameter);
    }

    let new_em : EMMultivariate = EMMultivariate {
        step: 0,
        dimension: dimension,
        x: x,
        parameters: parameters,
    };

    return new_em;
}

// ============================= UTIL FUNCTIONS =============================
fn compute_likelihood(x: Matrix, mean: Matrix, covariance: Matrix) -> Element {
    let sub = x - mean;
    let cov_determinant = covariance.determinant();
    let cov_inverse = covariance.cholesky().unwrap().inverse();
    
    let exponential_argument = -0.5 * (sub).transpose() * cov_inverse * (sub);

    let result = ( 1.0 / (2.0 * PI)) * ( 1.0 / cov_determinant) * (exponential_argument[(0, 0)]).exp();
    return result;
}

fn compute_joint_probability(prior: Element, likelihood: Element) -> Element {
    return prior * likelihood;
}

fn create_copy_with_value(mut matrix: Matrix, value: Element) -> Matrix {
    for element in matrix.iter_mut() {
        *element = value;
    }

    return matrix;
}

fn sum_of_all_posteriors(posteriors: &PosteriorsReturn) -> Element {
    let mut sum : Element = 0.0;
    for (_, element) in posteriors {
        sum = sum + *element;
    }

    return sum;
}