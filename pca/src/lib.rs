use nalgebra::{DVector, DMatrix, SymmetricEigen};

#[allow(dead_code)]
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
    eigen_values: Vector,
    eigen_vectors: Matrix,
}

pub struct PCA {
    covariance_type: CovarianceType,
    dimension: usize,
    points: Vec<Vector>,
    parameters: Option<Parameters>
}

impl PCA {

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

    fn compute_eigen(&self, covariance: Matrix) -> (Vector, Matrix) {
        let eigen = SymmetricEigen::new(covariance.clone());

        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        return (eigenvalues, eigenvectors);
    }

    fn compute_angle(&self) -> f64 {

        let eigen_vectors = &self.parameters.as_ref().unwrap().eigen_vectors;

        let dimension: usize = eigen_vectors.shape().1;
        let mut unit_vector: Vector = Vector::from_element(dimension, 0.0);
        unit_vector[(0, 0)] = 1.0;

        let u1 = eigen_vectors * &unit_vector;
        let u1_vector = Vector::from(u1);

        let angle = u1_vector[0].acos();
        return angle;
    }

    fn most_relevant(&self) -> usize {

        let mut max : Element = -1.0;
        let mut max_index : usize = 0;

        let eigen_values = &self.parameters.as_ref().unwrap().eigen_values;
        for (index, &value) in eigen_values.iter().enumerate() {
            if value > max {
                max = value;
                max_index = index;
            }
        }

        return max_index + 1;
    }

    fn discards_kaiser(&self) -> Vec<usize> {

        let mut discards: Vec<usize> = Vec::new();
        let eigen_values = &self.parameters.as_ref().unwrap().eigen_values;
        for (index, &value) in eigen_values.iter().enumerate() {
            if value < 1.0 { discards.push(index + 1) }
        }

        return discards;
    }

    pub fn compute_parameters(&mut self) {
        let mean = self.compute_mean();
        let covariance = self.compute_covariance(mean.clone());
        let eigen = self.compute_eigen(covariance.clone());
        
        let new_parameters = Parameters {
            means: mean,
            covariance: covariance,
            eigen_values: eigen.0,
            eigen_vectors: eigen.1, 
        };
        
        self.parameters = Some(new_parameters);
    }

    fn convert_point(&self, point: &Vector) -> Vector {

        let eigen_vectors : &Matrix = &self.parameters.as_ref().unwrap().eigen_vectors;
        let result : Vector = eigen_vectors.transpose() * point;
        return result;
    }

    pub fn print_parameters(&self) {
        if self.parameters.is_none() {
            println!("Please compute parameters first!");
            return;
        }

        let parameters = self.parameters.as_ref().unwrap();

        println!("Mean = {}", parameters.means);
        println!("Covariance = {}", parameters.covariance);
        println!("Eigenvalues = {}", parameters.eigen_values);
        println!("Eigenvectors = {}", parameters.eigen_vectors);

        // Extra information
        let angle = self.compute_angle();
        let angle_degrees = angle.to_degrees();
        println!("Angle = {} radians = {} º", angle, angle_degrees);
        let most_relevant = self.most_relevant();
        println!("Most significant eigenvector is λ_{}", most_relevant);
        let discards = self.discards_kaiser();
        println!("Kaiser discards: {:?} ( λ < 1 )", discards);

        println!();
        // Plot back the points
        for (index, point) in self.points.iter().enumerate() {
            let point_converted : Vector = self.convert_point(point);
            println!("x_eig^({}) = (transposed) {}", index + 1, point_converted.transpose());
        }

    }
}

pub fn build_pca(points: Vec<Vector>) -> PCA {
    let gaussian : PCA = PCA {
        covariance_type: CovarianceType::Sample,
        dimension: points[0].shape().0,
        points: points,
        parameters: None,
    };

    return gaussian;
}