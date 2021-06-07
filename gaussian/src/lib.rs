use nalgebra::{DVector, DMatrix};

enum CovarianceType {
    Population,
    Sample,
}

pub type Element = f64;
pub type Vector = DVector<Element>;
pub type Matrix = DMatrix<Element>;

pub struct Parameters {
    means: Vector,
    covariance: Matrix,
}

pub struct Gaussian {
    covariance_type: CovarianceType,
    dimension: usize,
    points: Vec<Vector>,
    parameters: Option<Parameters>
}

impl Gaussian {

    fn compute_mean(&self) -> Vector {
        let mut mean = Vector::from_element(self.dimension, 0.0);

        for point in self.points.iter() { mean = mean + point }
        mean = mean / self.points.len() as Element;

        return mean;
    }

    fn compute_covariance(&self, mean: Vector) -> Matrix {
        let mut covariance = Matrix::from_element(self.dimension, self.dimension, 0.0);

        for row in 0..self.dimension {
            for collumn in 0..self.dimension {

                let mut value = 0.0;
                for point in &self.points {

                    let tmp_value = (point[row] - mean[row]) * (point[collumn] - mean[collumn]);
                    value = value + tmp_value;
                }

                covariance[(row, collumn)] = value;
            }
        }

        match self.covariance_type {
            CovarianceType::Population => covariance = covariance / (self.points.len() as f64),
            CovarianceType::Sample => covariance = covariance / (self.points.len() as f64 - 1.0),
        }

        return covariance;
    }

    pub fn compute_parameters(&mut self) {
        let mean = self.compute_mean();
        let covariance = self.compute_covariance(mean.clone());

        let new_parameters = Parameters {
            means: mean,
            covariance: covariance,
        };

        self.parameters = Some(new_parameters);
    }

    pub fn print_parameters(&self) {
        if self.parameters.is_none() {
            println!("Please compute parameters first!");
            return;
        }

        println!("Mean = {}", self.parameters.as_ref().unwrap().means);
        println!("Covariance = {}", self.parameters.as_ref().unwrap().covariance);
    }
}

pub fn build_gaussian(points: Vec<Vector>) -> Gaussian {
    let gaussian : Gaussian = Gaussian {
        covariance_type: CovarianceType::Sample,
        dimension: points[0].shape().0,
        points: points,
        parameters: None,
    };

    return gaussian;
}