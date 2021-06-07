use nalgebra::{DVector, DMatrix};
use std::f64::consts::PI;

pub type Element = f64;
pub type Vector = DVector<Element>;
pub type Matrix = DMatrix<Element>;

pub struct Gaussian {
    mean: Vector,
    covariance: Matrix,
}

impl Gaussian {

    pub fn run_point(&self, point: Vector) -> f64 {
        let point_value = compute_likelihood(&point, &self.mean, &self.covariance);
        println!("Point has {} => {}", point_value, point);
        println!();

        return point_value;
    }
}

pub fn build_gaussian(mean: Vector, covariance: Matrix) -> Gaussian {
    let gaussian : Gaussian = Gaussian {
        mean: mean,
        covariance: covariance,
    };

    return gaussian;
}

fn compute_likelihood(point: &Vector, mean: &Vector, covariance: &Matrix) -> Element {
    let dimension = point.shape().0 as f64;

    let sub = point - mean;
    let cov_determinant = covariance.determinant();
    let cov_inverse = covariance.clone().cholesky().unwrap().inverse();
    
    let exponential_argument = -0.5 * (sub).transpose() * cov_inverse * (sub);

    let result = ((2.0 * PI).powf(- dimension / 2.0)) * ( cov_determinant.powf(-0.5)) * (exponential_argument[(0, 0)]).exp();
    return result;
}