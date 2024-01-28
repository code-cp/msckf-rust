use nalgebra as na; 

use crate::my_types::*; 

/// slam book eq. 2.4 
pub fn skew(v: na::Vector3<f64>) -> Matrix3d {
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
        q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3], 2.*q[1]*q[2] - 2.*q[0]*q[3], 2.*q[1]*q[3] + 2.*q[0]*q[2],
        2.*q[1]*q[2] + 2.*q[0]*q[3], q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3], 2.*q[2]*q[3] - 2.*q[0]*q[1],
        2.*q[1]*q[3] - 2.*q[0]*q[2], 2.*q[2]*q[3] + 2.*q[0]*q[1], q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3],
      )
}