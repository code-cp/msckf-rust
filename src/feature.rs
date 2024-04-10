use nalgebra as na; 

use crate::camera::CameraState;
use crate::my_types::*;

#[derive(Clone, Copy, Debug)]
pub struct Feature {
    pub point: Vector2d,
    pub id: TrackId,
}

#[derive(Clone, Debug)]
pub struct TrackPoint {
    pub coordinates: [Vector2d; 2],
    pub normalized_coordinates: [Vector2d; 2],
    pub frame_number: usize,
}

#[derive(Clone, Debug)]
pub struct Track {
    pub points: Vec<TrackPoint>,
    pub id: TrackId,
    pub last_seen: usize,
}

impl Track {
    pub fn new(
        features: [Feature; 2],
        normalized_coordinates: [Vector2d; 2],
        last_seen: usize,
        frame_number: usize,
    ) -> Self {
        assert_eq!(features[0].id, features[1].id);
        Track {
            points: vec![TrackPoint {
                coordinates: [features[0].point, features[1].point],
                normalized_coordinates,
                frame_number,
            }],
            id: features[0].id,
            last_seen,
        }
    }
}

// The method below does not work well 
// Algorithm from the book Computer Vision: Algorithms and Applications
// by Richard Szeliski. Chapter 7.1 Triangulation, page 345.
//
// There are many algorithms for triangulation using N cameras. This one is
// probably the simplest to differentiate wrt to all the pose variables.
// However, it has a particular weakness in that it ignores the fact that the
// fixed transformation between the stereo cameras is known, and instead treats
// all the camera rays as equal. Using this triangulation function may degrade
// quality of the visual updates considerably.
//
// NOTE This function is heavily based on the HybVIO implementation here:
//   <https://github.com/SpectacularAI/HybVIO/blob/main/src/odometry/triangulation.cpp>
//   (see `triangulateLinear()`)
// pub fn triangulate(
//     normalized_coordinates: &[[Vector2d; 2]],
//     camera_poses: &[&CameraState],
// ) -> Option<Vector3d> {
//     assert_eq!(normalized_coordinates.len(), camera_poses.len());

//     let mut s = Matrix3d::zeros();
//     let mut t = Vector3d::zeros();

//     for i in 0..normalized_coordinates.len() {
//         let camera_poses = camera_poses[i].convert_to_stereo_poses_vec(); 
//         for j in 0..2 {
//             let pose = &camera_poses[j]; 
//             let ip = &normalized_coordinates[i][j];
//             let ip = Vector3d::new(ip[0], ip[1], 1.);
//             // NOTE, need to use normalize, since vj is a unit vector for direction 
//             let vj = pose.orientation * ip.normalize();
//             let a = Matrix3d::identity() - vj * vj.transpose();
//             s += a;
//             t += a * pose.position;
//         }
//     }

//     let inv_s = s.try_inverse()?;
//     if inv_s[(0,0)].is_nan() {
//         return None; 
//     }
//     let position = inv_s * t;

//     Some(position)
// }

macro_rules! get_rotation_from_trans {
    ($mat:expr) => {
        $mat.fixed_view::<3,3>(0, 0)
    };
}

macro_rules! get_position_from_trans {
    ($mat:expr) => {
        $mat.fixed_view::<3,1>(0, 3)
    };
}

macro_rules! convert_vec3_to_vec2 {
    ($vec:expr) => {
        Vector2d::new($vec[0] / $vec[2], $vec[1] / $vec[2])
    };
}

macro_rules! convert_xy_to_xy1 {
    ($vec:expr) => {
        Vector3d::new($vec[0], $vec[1], 1.)
    };
}

macro_rules! convert_xyz_to_xyz1 {
    ($vec:expr) => {
        Vector4d::new($vec[0], $vec[1], $vec[2], 1.)
    };
}

/// Compute the cost of the camera observations
/// The ith measurement of the feature j in ci frame
fn get_cost(ci_trans_c0: &Matrix4d, position: &Vector3d, z: &Vector2d) -> f64 {

    // Compute hi1, hi2, and hi3 as Equation (37) in stereo msckf paper 
    let h = get_rotation_from_trans!(ci_trans_c0) * convert_xy_to_xy1!(position) + position[2] * get_position_from_trans!(ci_trans_c0); 

    if h[2] < 1e-4 {
        return f64::MAX; 
    }

    // Predict the feature observation in ci frame
    let z_hat = convert_vec3_to_vec2!(h);

    (z_hat - z).norm().powi(2)
}

// Compute the Jacobian of the camera observation
fn get_jacobian(ci_trans_c0: &Matrix4d, position: &Vector3d, z: &Vector2d, jacobian: &mut Matrixd, residual: &mut Vector2d, weight: &mut f64) {
    // Compute hi1, hi2, and hi3 as Equation (37) in stereo msckf paper 
    let h = get_rotation_from_trans!(ci_trans_c0) * convert_xy_to_xy1!(position) + position[2] * get_position_from_trans!(ci_trans_c0); 

    // Compute the Jacobian
    let mut w = Matrix3d::zeros(); 
    w.fixed_view_mut::<3, 2>(0, 0).copy_from(&ci_trans_c0.fixed_view::<3, 2>(0, 0)); 
    w.fixed_view_mut::<3, 1>(0, 2).copy_from(&ci_trans_c0.fixed_view::<3, 1>(0, 3));

    // let mut jacobian = Matrixd::zeros(2, 3); 
    jacobian.row_mut(0).copy_from(&(1./h[2] * w.row(0) - h[0]/h[2].powi(2) * w.row(2)));
    jacobian.row_mut(1).copy_from(&(1./h[2] * w.row(1) - h[1]/h[2].powi(2) * w.row(2))); 

    // Compute the residual
    let z_hat = convert_vec3_to_vec2!(position); 
    *residual = z_hat - z; 

    // Compute the weight based on the residual
    let e = residual.norm(); 
    let huber_epsilon = 0.01; 
    if e <= huber_epsilon {
        *weight = 1.0;
    } else {
        *weight = huber_epsilon / (2.*e); 
    }
}

pub fn triangulate(
    normalized_coordinates: &[[Vector2d; 2]],
    camera_poses: &[&CameraState],
) -> Option<Vector3d> {
    assert_eq!(normalized_coordinates.len(), camera_poses.len());

    // get initial guess 
    let cam1_trans_cam0 = camera_poses[0].cam1_trans_cam0.clone();
    let z1 = &normalized_coordinates[0][0];
    let z1 = Vector3d::new(z1[0], z1[1], 1.);

    let z2 = &normalized_coordinates[0][1];

    let m = cam1_trans_cam0.orientation * z1; 
    let mut a = Vector2d::zeros(); 
    a[0] = m[0] - z2[0]*m[2]; 
    a[1] = m[1] - z2[1]*m[2]; 

    let mut b = Vector2d::zeros();
    b[0] = z2[0] * cam1_trans_cam0.position[2] - cam1_trans_cam0.position[0];
    b[1] = z2[1] * cam1_trans_cam0.position[2] - cam1_trans_cam0.position[1];

    // feature position in camera frame 
    let mut position = Vector3d::zeros(); 
    let depth = (a.transpose() * a).try_inverse().unwrap() * a.transpose() * b; 
    position[0] = z1[0] * depth[(0,0)]; 
    position[1] = z1[1] * depth[(0,0)];
    position[2] = depth[(0,0)];

    // Apply Levenberg-Marquart method to solve for the 3d position.
    let mut solution = convert_xy_to_xy1!(position) / position[2]; 
    
    let initial_damping = 1e-3; 
    let mut lambda = initial_damping;

    let tolerance = 1e-3; 

    let outer_loop_max_iteration = 5; 
    let inner_loop_max_iteration = 5; 
    let estimation_precision = 5e-7; 

    let mut inner_loop_cntr = 0; 
    let mut outer_loop_cntr = 0; 
    let mut is_cost_reduced = false; 
    let mut delta_norm = f64::MAX; 

    // Compute the initial cost
    let mut total_cost = 0.0; 
    
    let mut camera_poses_relative = Vec::new(); 
    let camera0_pose = camera_poses[0].get_transformation_matrix(); 
    for pose in camera_poses {
        let stereo_pose = pose.convert_to_stereo_poses_vec();
        let mut cam0_pose = stereo_pose[0].get_transformation_matrix(); 
        let mut cam1_pose = stereo_pose[1].get_transformation_matrix();  

        cam0_pose = cam0_pose.try_inverse().unwrap() * camera0_pose; 
        cam1_pose = cam1_pose.try_inverse().unwrap() * camera0_pose; 

        camera_poses_relative.push(cam0_pose); 
        camera_poses_relative.push(cam1_pose); 
    }

    for (i, stereo_pose) in camera_poses_relative.chunks(2).enumerate() {
        for j in 0..2 {
            total_cost += get_cost(&stereo_pose[j], &solution, &normalized_coordinates[i][j]); 
        }
    }

    // Outer loop
    while outer_loop_cntr < outer_loop_max_iteration && delta_norm > estimation_precision {
        let mut a = Matrix3d::zeros(); 
        let mut b = Vector3d::zeros(); 

        for (i, stereo_pose) in camera_poses_relative.chunks(2).enumerate() {
            for j in 0..2 {
                let mut jacobian = Matrixd::zeros(2, 3); 
                let mut residual = Vector2d::zeros(); 
                let mut weight = 0.; 
    
                get_jacobian(&stereo_pose[0], &solution, &normalized_coordinates[i][j], &mut jacobian, &mut residual, &mut weight); 
            
                a += weight.powi(2) * jacobian.transpose() * jacobian.clone(); 
                b += weight.powi(2) * jacobian.transpose() * residual; 
            } 
        }

        // Inner loop
        // Solve for the delta that can reduce the total cost
        while inner_loop_cntr < inner_loop_max_iteration && !is_cost_reduced {
            let damper = lambda * Matrix3d::identity(); 

            // svd solve Solves the system self * x = b where self is the decomposed matrix and x the unknown.
            let delta = na::linalg::SVD::new(a + lambda * damper, true, true)
                .solve(&b, tolerance)
                .unwrap_or(Vector3d::zeros());
            let new_solution = solution - delta;
            delta_norm = delta.norm(); 

            let mut new_cost = 0.; 
            for (i, stereo_pose) in camera_poses_relative.chunks(2).enumerate() {
                for j in 0..2 {
                    new_cost += get_cost(&stereo_pose[j], &new_solution, &normalized_coordinates[i][j]); 
                }
            }

            if new_cost < total_cost {
                is_cost_reduced = true; 
                solution = new_solution; 
                lambda = if lambda / 10. > 1e-10 {lambda / 10.} else {1e-10}; 
            } else {
                is_cost_reduced = false; 
                lambda = if lambda * 10. < 1e12 {lambda * 10.} else {1e12}; 
            }

            inner_loop_cntr += 1; 
        }

        inner_loop_cntr = 0; 
        outer_loop_cntr += 1; 
    }

    let final_position = Vector3d::new(
        solution[0] / solution[2], 
        solution[1] / solution[2], 
        1.0 / solution[2], 
    ); 

    for pose in camera_poses_relative.iter() {
        let position = *pose * convert_xyz_to_xyz1!(final_position);
        if position[2] < 1e-3 {
            return None; 
        } 
    }

    // feature position in world frame 
    let final_position = camera_poses[0].orientation * final_position + camera_poses[0].position; 

    Some(final_position)
}