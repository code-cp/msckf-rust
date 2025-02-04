use nalgebra as na;

use crate::my_types::*;

/// slam book eq. 2.4
pub fn skew(v: &na::Vector3<f64>) -> Matrix3d {
    let mut ss = Matrix3d::zeros();
    ss[(0, 1)] = -v[2];
    ss[(0, 2)] = v[1];
    ss[(1, 0)] = v[2];
    ss[(1, 2)] = -v[0];
    ss[(2, 0)] = -v[1];
    ss[(2, 1)] = v[0];
    ss
}

/// ref Quaternion kinematics for the error-state Kalman filter
/// eq. 115
/// quaternion format wxyz 
pub fn to_rotation_matrix(q: Vector4d) -> Matrix3d {
    Matrix3d::new(
        q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3],
        2. * q[1] * q[2] - 2. * q[0] * q[3],
        2. * q[1] * q[3] + 2. * q[0] * q[2],
        2. * q[1] * q[2] + 2. * q[0] * q[3],
        q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3],
        2. * q[2] * q[3] - 2. * q[0] * q[1],
        2. * q[1] * q[3] - 2. * q[0] * q[2],
        2. * q[2] * q[3] + 2. * q[0] * q[1],
        q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3],
    )
}

/// Derivatives of the rotation matrix w.r.t. the quaternion
/// TODO how to derive
pub fn drot_mat_dq(q: Vector4d) -> [Matrix3d; 4] {
    [
        Matrix3d::new(
            2. * q[0],
            -2. * q[3],
            2. * q[2],
            2. * q[3],
            2. * q[0],
            -2. * q[1],
            -2. * q[2],
            2. * q[1],
            2. * q[0],
        ),
        Matrix3d::new(
            2. * q[1],
            2. * q[2],
            2. * q[3],
            2. * q[2],
            -2. * q[1],
            -2. * q[0],
            2. * q[3],
            2. * q[0],
            -2. * q[1],
        ),
        Matrix3d::new(
            -2. * q[2],
            2. * q[1],
            2. * q[0],
            2. * q[1],
            2. * q[2],
            2. * q[3],
            -2. * q[0],
            2. * q[3],
            -2. * q[2],
        ),
        Matrix3d::new(
            -2. * q[3],
            -2. * q[0],
            2. * q[1],
            2. * q[0],
            -2. * q[3],
            2. * q[2],
            2. * q[1],
            2. * q[2],
            2. * q[3],
        ),
    ]
}

pub fn so3_exp(phi: &Vector3d) -> Matrix3d {
    let theta = phi.norm();
    // void nan 
    let n = phi / (theta + 1e-4);
    let exp_phi_skew = theta.cos() * Matrix3d::identity()
        + (1.0 - theta.cos()) * n * n.transpose()
        + theta.sin() * skew(&n);
    exp_phi_skew
}

/// slam book eq. 3.27
fn calculate_j(theta: f64, n: &Vector3d) -> Matrix3d {
    let i3 = Matrix3d::identity(); 
    // need to check value of theta 
    // otherwise this function will be super slow 
    if theta > 1e-6 {
        (theta.sin() / theta) * i3 
        + (1.0 - theta.sin() / theta) * n * n.transpose() 
        + (1.0 - theta.cos()) / theta * skew(n)
    } else {
        i3 
    }
}

/// slam book p. 65 
pub fn se3_exp(v: &Vector6d) -> Matrix4d {
    let rho = Vector3d::new(v[0], v[1], v[2]); 
    let phi = Vector3d::new(v[3], v[4], v[5]); 

    let theta = phi.norm();
    // avoid dividing by zero 
    let n = phi / (theta + 1e-5); 
    let exp_phi_skew = theta.cos() * Matrix3d::identity() 
        + (1.0 - theta.cos()) * n * n.transpose() 
        + theta.sin() * skew(&n);  

    let j = calculate_j(theta, &n);
    
    let mut se3 = Matrix4d::zeros(); 
    se3.fixed_view_mut::<3,3>(0,0).copy_from(&exp_phi_skew);
    se3.fixed_view_mut::<3,1>(0,3).copy_from(&(j*rho)); 
    se3[(3,3)] = 1.; 

    se3
}

pub fn hnormalize(p: Vector3d) -> Option<Vector2d> {
    if p[2] <= 0. {
        return None;
    }
    Some(Vector2d::new(p[0] / p[2], p[1] / p[2]))
}

pub fn odot_operator(x: &Vector4d) -> Matrixd {
    let x = Vector3d::new(x[0], x[1], x[2]); 
    let mut result = Matrixd::zeros(4, 6);
    result
        .view_mut((0, 0), (3, 3))
        .copy_from(&Matrix3d::identity());
    result
        .view_mut((0, 3), (3, 3))
        .copy_from(&(-skew(&x)));
    result
}

/// ref <https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py>
pub fn rotation_matrix_to_angle_axis(rotation: &Matrix3d) -> Vector3d {
    // Extract the rotation axis
    let axis = Vector3d::new(rotation[(2, 1)] - rotation[(1, 2)],
                            rotation[(0, 2)] - rotation[(2, 0)],
                            rotation[(1, 0)] - rotation[(0, 1)]);
    let axis = if axis.norm() > 1e-2 {axis.normalize()} else {axis}; 

    // Compute the angle
    let angle_cos = (rotation[(0, 0)] + rotation[(1, 1)] + rotation[(2, 2)] - 1.0) / 2.0;
    let angle = angle_cos.min(1.0).max(-1.0).acos(); // Angle in radians

    axis*angle
}

/// ref <https://github.com/raulmur/ORB_SLAM2/issues/766#issuecomment-516910030>
pub fn rotation_to_euler(rotation: &Matrix3d) -> Vector3d {
    let m00 = rotation[(0, 0)];
    let m02 = rotation[(0, 2)];
    let m10 = rotation[(1, 0)];
    let m11 = rotation[(1, 1)];
    let m12 = rotation[(1, 2)];
    let m20 = rotation[(2, 0)];
    let m22 = rotation[(2, 2)];

    let (bank, attitude, heading) = if m10 > 0.998 { // singularity at north pole
        (0.0, std::f64::consts::FRAC_PI_2, m02.atan2(m22))
    } else if m10 < -0.998 { // singularity at south pole
        (0.0, -std::f64::consts::FRAC_PI_2, m02.atan2(m22))
    } else {
        (
            (-m12).atan2(m11),
            m10.asin(),
            (-m20).atan2(m00),
        )
    };

    Vector3d::new(bank, attitude, heading)
}

pub fn matrix_to_quaternion(matrix: &Matrix3d) -> na::Quaternion<f64> {
    let trace = matrix[(0, 0)] + matrix[(1, 1)] + matrix[(2, 2)];

    if trace > 0.0 {
        let s = (0.5 / trace.sqrt()).sqrt();
        na::Quaternion::new(
            (matrix[(2, 1)] - matrix[(1, 2)]) * s,
            (matrix[(0, 2)] - matrix[(2, 0)]) * s,
            (matrix[(1, 0)] - matrix[(0, 1)]) * s,
            0.25 / s,
        )
    } else if matrix[(0, 0)] > matrix[(1, 1)] && matrix[(0, 0)] > matrix[(2, 2)] {
        let s = (2.0 * (1.0 + matrix[(0, 0)] - matrix[(1, 1)] - matrix[(2, 2)])).sqrt();
        na::Quaternion::new(
            0.25 * s,
            (matrix[(0, 1)] + matrix[(1, 0)]) / s,
            (matrix[(0, 2)] + matrix[(2, 0)]) / s,
            (matrix[(2, 1)] - matrix[(1, 2)]) / s,
        )
    } else if matrix[(1, 1)] > matrix[(2, 2)] {
        let s = (2.0 * (1.0 + matrix[(1, 1)] - matrix[(0, 0)] - matrix[(2, 2)])).sqrt();
        na::Quaternion::new(
            (matrix[(0, 1)] + matrix[(1, 0)]) / s,
            0.25 * s,
            (matrix[(1, 2)] + matrix[(2, 1)]) / s,
            (matrix[(0, 2)] - matrix[(2, 0)]) / s,
        )
    } else {
        let s = (2.0 * (1.0 + matrix[(2, 2)] - matrix[(0, 0)] - matrix[(1, 1)])).sqrt();
        na::Quaternion::new(
            (matrix[(0, 2)] + matrix[(2, 0)]) / s,
            (matrix[(1, 2)] + matrix[(2, 1)]) / s,
            0.25 * s,
            (matrix[(1, 0)] - matrix[(0, 1)]) / s,
        )
    }
}

pub fn hstack_mat(mat_a: &Matrixd, mat_b: &Matrixd) -> Matrixd {
    let num_rows_c = mat_a.nrows() + mat_b.nrows();
    let num_cols_c = mat_a.ncols();
    let mut c = Matrixd::zeros(num_rows_c, num_cols_c);
    c.view_mut((0, 0), (mat_a.nrows(), num_cols_c)).copy_from(&mat_a); 
    c.view_mut((mat_a.nrows(), 0), (mat_b.nrows(), num_cols_c)).copy_from(&mat_b); 
    c
}

pub fn hstack_vec(mat_a: &Vectord, mat_b: &Vectord) -> Vectord {
    let num_rows_c = mat_a.nrows() + mat_b.nrows();
    let num_cols_c = mat_a.ncols();
    let mut c = Vectord::zeros(num_rows_c);
    c.view_mut((0, 0), (mat_a.nrows(), num_cols_c)).copy_from(&mat_a); 
    c.view_mut((mat_a.nrows(), 0), (mat_b.nrows(), num_cols_c)).copy_from(&mat_b); 
    c
}