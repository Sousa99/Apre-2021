use nalgebra::{DVector, DMatrix};

pub type Element = f64;
pub type Vector = DVector<Element>;
pub type Matrix = DMatrix<Element>;

#[derive(Copy, Clone)]
enum ErrorFunctionType {
    SquaredError,
}

#[derive(Copy, Clone)]
enum ActivationFunctionType {
    Linear,
}

pub struct ClosedForm {
    error_function: ErrorFunctionType,
    activation_function: ActivationFunctionType,
    points: Matrix,
    target: Vector,
    lambda: Element,
}

impl ClosedForm {

    pub fn compute_weights(&self) -> Vector {

        // Auxiliary values
        let points_transpose = self.points.clone().transpose();
        println!("X^T = {}", points_transpose);
        let squared_points = points_transpose.clone() * self.points.clone();
        println!("(X^T * X) = {}", squared_points);
        let dimension : (usize, usize) = squared_points.clone().shape();
        let lambda_matrix = squared_points + self.lambda * Matrix::identity(dimension.0, dimension.1);
        println!("(X^T * X + λI) = {}", lambda_matrix);
        let inverse = lambda_matrix.cholesky().unwrap().inverse();
        println!("(X^T * X + λI)^-1 = {}", inverse);

        let weights = match (self.error_function, self.activation_function) {
            (ErrorFunctionType::SquaredError, ActivationFunctionType::Linear) => inverse * points_transpose.clone() * self.target.clone(),
        };

        return weights;
    }
}

pub fn build_closed_form(points: Matrix, target: Vector, lambda: Element) -> ClosedForm {
    let closed_form : ClosedForm = ClosedForm {
        error_function: ErrorFunctionType::SquaredError,
        activation_function: ActivationFunctionType::Linear,
        points: points,
        target: target,
        lambda: lambda,
    };

    return closed_form;
}