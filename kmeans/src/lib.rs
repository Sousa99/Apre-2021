use nalgebra::{DVector, DMatrix};
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{PointMarker, PointStyle};
use plotlib::view::ContinuousView;

pub type Element = f64;
pub type Vector = DVector<Element>;
pub type Matrix = DMatrix<Element>;

pub struct KMeans {
    export_image: bool,
    x: Vec<Vector>,
    cluster_centers: Vec<Vector>,
}

impl KMeans {

    pub fn do_loop_iterations(&mut self, iterations: usize) {
        for iteration in 0..iterations {
            let (points_in_clusters, distance) = self.compute_clusters();
            
            // Export before cluster centers update
            if self.export_image {
                let file_name_pre = format!("./results/iteration-{}-pre.svg", iteration + 1);
                export_kmeans_fig(file_name_pre, self.cluster_centers.clone(), points_in_clusters.clone());
            }
            
            let converged = self.update_cluster_centers(points_in_clusters.clone());
            if converged {
                println!("Result converged after {} iterations!", iteration);
                println!();
                break;
            }

            println!(" ===================== Iteration {} =====================", iteration);
            println!("Distance Matrix: (using squared distance) {}", distance);
            for (cluster_index, cluster_center) in self.cluster_centers.iter().enumerate() {
                println!("C{}: {}", cluster_index + 1, cluster_center);
            }

            // Export after cluster centers update
            if self.export_image {
                let file_name_pre = format!("./results/iteration-{}-pos.svg", iteration + 1);
                export_kmeans_fig(file_name_pre, self.cluster_centers.clone(), points_in_clusters.clone());
            }
        }

        println!("Intra-Cluster distance = {} (using squared distance)", self.compute_intra_cluster_distance());
        println!("Inter-Cluster distance = {} (using squared distance)", self.compute_inter_cluster_distance());

    }

    fn compute_clusters(&self) -> (Vec<Vec<Vector>>, Matrix) {
        let mut clusters : Vec<Vec<Vector>> = Vec::new();
        let number_clusters : usize = self.cluster_centers.len();
        let mut distances_vec : Vec<Element> = Vec::new();

        // Initialize every cluster as an empty list
        for _ in 0..number_clusters { clusters.push(Vec::new()); }

        // Place points in correct clusters
        for point in &self.x {

            let mut min_cluster : Option<(usize, f64)> = None;
            for (cluster_index, cluster_center) in self.cluster_centers.iter().enumerate() {

                let distance_to_cluster = compute_distance_squared(point.clone(), cluster_center.clone());
                distances_vec.push(distance_to_cluster);
                match min_cluster {
                    Some((_, min_distance)) if distance_to_cluster < min_distance => min_cluster = Some((cluster_index, distance_to_cluster)),
                    Some(_) => {},
                    None => min_cluster = Some((cluster_index, distance_to_cluster)),
                }
            }

            if min_cluster.is_some() {
                let (cluster_index, _) = min_cluster.unwrap();
                clusters[cluster_index].push(point.clone());
            }
        }

        let distance_matrix : Matrix = Matrix::from_vec(self.cluster_centers.len(), self.x.len(), distances_vec);

        return (clusters, distance_matrix);
    }

    fn compute_intra_cluster_distance(&self) -> f64 {
        let mut distance : f64 = 0.0;
        let (points_in_clusters, _) = self.compute_clusters();

        for (cluster, points) in self.cluster_centers.iter().zip(points_in_clusters.iter()) {
            for point in points { distance = distance + compute_distance_squared(point.clone(), cluster.clone()) }
        }

        return distance;
    }

    fn compute_inter_cluster_distance(&self) -> f64 {
        let mut distance : f64 = 0.0;
        let number_clusters : usize = self.cluster_centers.len();

        for index_a in 0..number_clusters {
            let cluster_a = self.cluster_centers.get(index_a).unwrap();
            for index_b in 0..number_clusters {
                let cluster_b = self.cluster_centers.get(index_b).unwrap();
                distance = distance + compute_distance_squared(cluster_a.clone(), cluster_b.clone());
            }
        }

        return distance / number_clusters.pow(2) as f64;
    }

    fn update_cluster_centers(&mut self, points_in_clusters: Vec<Vec<Vector>>) -> bool {
        let mut converged : bool = true;
        let mut new_cluster_centers : Vec<Vector> = Vec::new();

        for (cluster_set, current_center) in points_in_clusters.into_iter().zip(self.cluster_centers.iter()) {
            let number_points : usize = cluster_set.len();
            let mut sum : Vector = create_copy_with_value(cluster_set[0].clone(), 0.0);

            for point in cluster_set {
                sum = sum + point;
            }

            let result = sum / (number_points as f64);
            if result != current_center.clone() { converged = false; }
            new_cluster_centers.push(result);
        }

        self.cluster_centers = new_cluster_centers;
        return converged;
    }
}

pub fn build_kmeans(export_image: bool, x: Vec<Vector>, cluster_centers: Vec<Vector>) -> KMeans {
    let new_kmeans : KMeans = KMeans {
        export_image: export_image,
        x: x,
        cluster_centers: cluster_centers,
    };

    return new_kmeans;
}

fn create_copy_with_value(mut vector: Vector, value: Element) -> Vector {
    for element in vector.iter_mut() {
        *element = value;
    }

    return vector;
}

pub fn compute_distance_squared(vector_1: Vector, vector_2: Vector) -> f64 {
    let mut distance : f64 = 0.0;

    for (point_1, point_2) in vector_1.iter().zip(vector_2.iter()) {
        distance = distance + (point_1 - point_2).powi(2);
    }

    return distance;
}


type PointPlot = (Element, Element);
fn export_kmeans_fig(file_name: String, cluster_centers: Vec<Vector>, points_in_clusters: Vec<Vec<Vector>>) {
    let colors : Vec<&str> = vec!("#35C788", "#DD3355", "#3388DD", "#101010");
    
    let mut datasets : Vec<(Vec<PointPlot>, PointStyle)> = Vec::new();
    // Deal with clusters
    for (cluster_index, cluster_center) in convert_vec_vectors_to_vec_points(cluster_centers).into_iter().enumerate() {
        let points_vec : Vec<PointPlot> = vec!(cluster_center);
        let color : &str = colors[cluster_index];
        let points_style : PointStyle = PointStyle::new()
            .marker(PointMarker::Square).colour(color);

        datasets.push((points_vec, points_style));
    }
    // Deal with points in clusters
    for (cluster_index, points_in_cluster) in points_in_clusters.into_iter().enumerate() {
        let points_vec : Vec<PointPlot> = convert_vec_vectors_to_vec_points(points_in_cluster);
        let color : &str = colors[cluster_index];
        let points_style : PointStyle = PointStyle::new().colour(color);

        datasets.push((points_vec, points_style));
    }

    save_chart(file_name, datasets);
}

fn convert_vec_vectors_to_vec_points(initial_vector: Vec<Vector>) -> Vec<PointPlot> {

    let mut final_vector : Vec<PointPlot> = Vec::new();
    for vector in initial_vector {
        let new_point : PointPlot = (vector[0], vector[1]);
        final_vector.push(new_point);
    }

    return final_vector;
}

fn save_chart(file_name: String, datasets: Vec<(Vec<PointPlot>, PointStyle)>) {
    let mut view : ContinuousView = ContinuousView::new();

    let (min_x, max_x, min_y, max_y) = get_min_and_max_from_dataset(&datasets);
    let distance_x = (max_x - min_x) * 0.05;
    let distance_y = (max_y - min_y) * 0.05;

    for dataset in datasets.into_iter() {
        let (points, point_style) = dataset;
        let set : Plot = Plot::new(points).point_style(point_style);
        view = view.add(set);
    }

    view = view.x_range(min_x - distance_x, max_x + distance_x)
        .y_range(min_y - distance_y, max_y + distance_y)
        .x_label("x1")
        .y_label("x2");

    Page::single(&view).save(file_name).unwrap();
}

fn get_min_and_max_from_dataset(datasets: &Vec<(Vec<PointPlot>, PointStyle)>) -> (Element, Element, Element, Element) {
    let mut min_x : Option<Element> = None;
    let mut min_y : Option<Element> = None;
    let mut max_x : Option<Element> = None;
    let mut max_y : Option<Element> = None;

    for set in datasets {
        let (points, _) = set;
        for &point in points {
            let (point_x, point_y) = point;

            match min_x {
                Some(value) if point_x < value => min_x = Some(point_x),
                Some(_) => {},
                None => min_x = Some(point_x),
            }
            match min_y {
                Some(value) if point_y < value => min_y = Some(point_y),
                Some(_) => {},
                None => min_y = Some(point_y),
            }
            match max_x {
                Some(value) if point_x > value => max_x = Some(point_x),
                Some(_) => {},
                None => max_x = Some(point_x),
            }
            match max_y {
                Some(value) if point_y > value => max_y = Some(point_y),
                Some(_) => {},
                None => max_y = Some(point_y),
            }
        }
    }

    return (min_x.unwrap(), max_x.unwrap(), min_y.unwrap(), max_y.unwrap());
}