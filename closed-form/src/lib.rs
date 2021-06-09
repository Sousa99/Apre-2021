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
}

impl ClosedForm {

    pub fn compute_weights(&self) -> Vector {

        // Auxiliary values
        let points_transpose = self.points.clone().transpose();
        println!("X^T = {}", points_transpose);
        let squared_points = points_transpose.clone() * self.points.clone();
        println!("(X^T * X) = {}", squared_points);
        let inverse_squared_points = squared_points.cholesky().unwrap().inverse();
        println!("(X^T * X)^-1 = {}", inverse_squared_points);

        let weights = match (self.error_function, self.activation_function) {
            (ErrorFunctionType::SquaredError, ActivationFunctionType::Linear) => inverse_squared_points * points_transpose.clone() * self.target.clone(),
        };

        return weights;
    }
}

pub fn build_closed_form(points: Matrix, target: Vector) -> ClosedForm {
    let closed_form : ClosedForm = ClosedForm {
        error_function: ErrorFunctionType::SquaredError,
        activation_function: ActivationFunctionType::Linear,
        points: points,
        target: target,
    };

    return closed_form;
}