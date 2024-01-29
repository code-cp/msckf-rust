use nalgebra as na;
use opencv::core::Mat; 

use crate::my_types::*; 
use crate::config::*;
use crate::math::*;

const CAM_POS: usize = 0;
const CAM_ROT: usize = 3;
const CAM_SIZE: usize = 7;

const F_VEL: usize = 0; // Velocity.
const F_BGA: usize = 3; // [B]ias [G]yroscope [A]dditive.
const F_BAA: usize = 6; // [B]ias [A]ccelerometer [A]dditive.

// first camera pose 
const CAM0: usize = 9; // Start of camera poses. The most recent comes first.
const F_POS: usize = CAM0;
const F_ROT: usize = CAM0 + CAM_ROT;
const F_SIZE: usize = CAM0 + CAM_SIZE;

// Prediction noise.
const Q_A: usize = 0; // Accelerometer.
const Q_G: usize = 3; // Gyroscope.
const Q_BGA: usize = 6; // BGA drift.
const Q_BAA: usize = 9; // BAA drift.
const Q_SIZE: usize = 12;

pub struct KalmanFilter {
    state: Vectord, 
    state_len: usize, 
    state_cov: Matrixd, 

    gravity: Vector3d, 
    window_size: usize, 
    last_time: Option<f64>, 
}

impl KalmanFilter {
    pub fn new() -> Self {
        let config = CONFIG.get().unwrap(); 

        let gravity = Vector3d::new(0., 0., -config.gravity);
        let window_size = config.window_size;

        let state_len = 6;
        let state = na::DVector::zeros(state_len);
        let state_cov = na::DMatrix::zeros(state_len, state_len); 

        let set_diagonal_from_noise = | x: &mut Matrixd, start, len: usize, std_value: f64| {
            for i in start..(start + len) {
                x[(i, i)] = std_value.powi(2); 
            }
        };

        Self {
            state, 
            state_len,
            state_cov, 

            gravity, 
            window_size, 
            last_time: None, 
        }
    }

    pub fn get_velocity(&self) -> Vector3d {
        self.state.fixed_slice::<3, 1>(F_VEL, 0).into()
    }

    pub fn get_bga(&self) -> Vector3d {
        self.state.fixed_slice::<3, 1>(F_BGA, 0).into()
    }

    pub fn get_baa(&self) -> Vector3d {
        self.state.fixed_slice::<3, 1>(F_BAA, 0).into()
    }

    pub fn get_camera_position(&self, index: usize) -> Vector3d {
        self.state.fixed_slice::<3,1>(CAM0 + index * CAM_SIZE + CAM_POS, 0).into()
    }

    pub fn get_camera_rotation(&self, index: usize) -> Vector4d {
        self.state.fixed_slice::<4, 1>(CAM0 + index * CAM_SIZE + CAM_ROT, 0).into()
    }

    /// Initialize orientation based on the first accelometer sample.
    /// Based on <https://math.stackexchange.com/a/2313401>.
    /// ref to initializeGravityAndBias in stereo msckf 
    pub fn initialize_orientation(&mut self, acc: Vector3d) {
        let u = -self.gravity; 
        // This is the gravity in the IMU frame.
        let v = acc;

        let u_norm = u.norm(); 
        let v_norm = v.norm(); 
        let n = 1. / (u * v_norm + u_norm * v).norm(); 

        let w = n * (u_norm * v_norm + (u.transpose() * v)[(0, 0)]);
        let xyz = n * u.cross(&v); 

        // quaternion is represented by wxyz 
        // Hamilton convention
        // from world to IMU 
        self.state[F_ROT] = w; 
        self.state.fixed_slice_mut::<3, 1>(F_ROT + 1, 0).copy_from(&xyz); 
    }

    /// ref stereo msckf processModel
    pub fn predict(
        &mut self, 
        time: f64, 
        gyro: Vector3d, 
        acc: Vector3d, 
    ) {
        // initialize the orientation 
        if self.get_camera_rotation(0) == Vector3d::zeros() {
            self.initialize_orientation(acc);
        }

        // update dt 
        let dt = if let Some(last_time) = self.last_time {time - last_time} else { 0. }; 
        self.last_time = Some(time); 
        if dt <= 0. { return; }

        // remove the bias 
        let gyro = gyro - self.get_bga(); 
        let acc = acc - self.get_baa(); 

        // update imu position 
        let pos_new = self.get_camera_position(0) + self.get_velocity() * dt; 
        self.state.fixed_slice_mut::<3, 1>(F_POS, 0).copy_from(&pos_new); 

        // update imu orientation
        // ref to notes 
        // ref https://ahrs.readthedocs.io/en/latest/filters/angular.html
        // ref https://faculty.sites.iastate.edu/jia/files/inline-files/quaternion.pdf
        let mut omega = Matrix4d::zeros();
        omega.fixed_slice_mut::<3, 3>(1, 1).copy_from(&(-skew(gyro)));
        omega.fixed_slice_mut::<3, 1>(1, 0).copy_from(&gyro);
        omega.fixed_slice_mut::<1, 3>(0, 1).copy_from(&(-gyro.transpose()));
        // -0.5 instead of 0.5, because rotation is defined from world to imu 
        // not from imu to world 
        // ie still a local perturbation, but q is different 
        let omega = -0.5 * dt * omega.exp(); 
        let rot_new = omega * self.get_camera_rotation(0); 
        self.state.fixed_slice_mut::<4, 1>(F_ROT, 0).copy_from(&rot_new); 

        // update imu velocity 
        let last_q: Vector4d = self.get_camera_rotation(0).into(); 
        // from world to IMU 
        let i_r_w: Matrix3d = to_rotation_matrix(last_q);
        let vel_new = self.get_velocity() + (i_r_w.transpose() * acc + self.gravity) * dt;
        self.state.fixed_slice_mut::<3, 1>(F_VEL, 0).copy_from(&vel_new); 

    }

    pub fn construct_f_mat(&self, dt: f64, acc: &Vector3d, gyro: &Vector3d, omega: &Matrix4d) -> Matrixd {
        // Compute discrete transition and noise covariance matrix
        let mut f_mat: Matrixd = Matrixd::zeros(F_SIZE, F_SIZE); 
        let i3 = Matrix3d::identity(); 

        f_mat.fixed_slice_mut::<3, 3>(F_POS, F_POS).copy_from(&i3); 
        f_mat.fixed_slice_mut::<3, 3>(F_VEL, F_VEL).copy_from(&i3); 
        f_mat.fixed_slice_mut::<3, 3>(F_BGA, F_BGA).copy_from(&i3);
        f_mat.fixed_slice_mut::<3, 3>(F_BAA, F_BAA).copy_from(&i3); 

        // Derivatives of the velocity w.r.t. to the quaternion
        // TODO figure out derivation 
        let last_q: Vector4d = self.get_camera_rotation(0).into(); 
        let dr_dq = drot_mat_dq(last_q); 
        let mut y = Matrix34d::zeros(); 
        for i in 0..4 {
            y.fixed_slice_mut::<3, 1>(0, i).copy_from(&(dt * dr_dq[i].transpose() * acc));
        }
        // omega is the derivative of q dot wrt q, ref juan sola eq 200 
        f_mat.fixed_slice_mut::<3, 4>(F_VEL, F_ROT).copy_from(&(y * omega));

        // Derivatives of the quaternion w.r.t. itself
        f_mat.fixed_slice_mut::<4, 4>(F_ROT, F_ROT).copy_from(&omega);

        // Derivatives of the velocity w.r.t. to the gyro bias
        

        // Derivatives of the velocity w.r.t the acc. bias
        let i_r_w: Matrix3d = to_rotation_matrix(last_q);
        f_mat.fixed_slice_mut::<3, 3>(F_VEL, F_BAA).copy_from(&(-dt * i_r_w.transpose()));

        f_mat
    }

}