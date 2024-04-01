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
    let n = phi / theta;
    let exp_phi_skew = theta.cos() * Matrix3d::identity()
        + (1.0 - theta.cos()) * n * n.transpose()
        + theta.sin() * skew(&n);
    exp_phi_skew
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
        .slice_mut((0, 0), (3, 3))
        .copy_from(&Matrix3d::identity());
    result
        .slice_mut((0, 3), (3, 3))
        .copy_from(&skew(&x));
    result
}
