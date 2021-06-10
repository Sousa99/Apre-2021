use nalgebra::{DVector};

pub type Element = f64;
pub type Vector = DVector<Element>;

pub struct PointInfo {
    point: Vector,
    target: Element,
    alpha: Element,
}

pub fn build_point_info(point: Vector, target: Element, alpha: Element) -> PointInfo {
    let point_info : PointInfo = PointInfo {
        point: point,
        target: target,
        alpha: alpha,
    };

    return point_info;
}

struct SVMParameters {
    weights: Vector,
    bias: Element,
}

pub struct SVM {
    points: Vec<PointInfo>,
    parameters: Option<SVMParameters>,
}

impl SVM {

    pub fn do_computations(&mut self) {

        let weights = self.compute_weights();
        let bias = self.compute_bias(&weights);
        let parameters = SVMParameters { weights: weights, bias: bias };
        self.parameters = Some(parameters);

        println!();
        self.compute_margin();
    }

    pub fn run_point(&self, point: &Vector) -> Element {

        let parameters : &SVMParameters = &self.parameters.as_ref().unwrap();
        let weights : &Vector = &parameters.weights;
        let bias : Element = parameters.bias;
        
        let value = (weights.transpose() * point)[(0, 0)] + bias;
        let result = value.signum();

        println!("o(x) = sgn( Σ t^(i) * a_i * x^t * x^(i) + b ) = sgn ( W^t * x + b )");
        println!("o(x) = sgn( {} ) = {}", value, result);

        return result;
    }

    fn compute_weights(&self) -> Vector {

        let mut valid_alphas : usize = 0;

        let dimension : usize = self.points[0].point.shape().0;
        let mut weights : Vector = Vector::from_element(dimension, 0.0);

        for point_info in self.points.iter() {
            if point_info.alpha == 0.0 { continue }

            weights = weights + point_info.target * point_info.alpha * &point_info.point;
            valid_alphas = valid_alphas + 1;
        }

        // ======================= PRINT COMPUTATIONS =======================

        println!("Support Vectors = {} (all that have α != 0)", valid_alphas);
        println!("Weight = w = Σ t^i * a_i * x^(i) = (transposed) {}", weights.transpose());

        return weights;
    }

    fn compute_bias(&self, weights: &Vector) -> Element {

        for point_info in self.points.iter() {
            if point_info.alpha == 0.0 { continue }

            let bias = - (weights.transpose() * &point_info.point)[(0, 0)] + point_info.target;

            println!("W^T * x^(i) + b = t^(i)");
            println!("Bias = b = {}", bias);
            return bias;
        }

        panic!("NO ALPHAS VALUE != 0");
    }

    fn compute_margin(&self) -> Element {

        let margin = 1.0 / euclidean_distance(&self.parameters.as_ref().unwrap().weights);
        println!("Margin = 1 / ||w||_2 = {}", margin);

        return margin;
    }
}

pub fn build_svm(points: Vec<PointInfo>) -> SVM {
    let svm : SVM = SVM {
        points: points,
        parameters: None,
    };

    return svm;
}

// ================================ AUXILIARY FUNCTIONS ================================

fn euclidean_distance(vector: &Vector) -> Element {
    
    let mut denominator: Element = 0.0;
    for value in vector.iter() { denominator = denominator + value.powi(2) }
    let square_root: Element = denominator.powf(0.5);

    return square_root;
}