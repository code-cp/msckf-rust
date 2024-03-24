use nalgebra as na;
use std::collections::HashMap;
use std::hash::Hash;

use crate::config::*;
use crate::math::*;
use crate::my_types::*;

static STATE_LEN: usize = 21;

pub struct CameraState {
    id: usize,
    // Takes a vector from the cam0 frame to the cam1 frame
    r_cam0_cam1: Matrix3d,
    t_cam0_cam1: Matrix3d,
    // Take a vector from the camera frame to world frame
    orientation: Matrix3d,
    position: Vector3d,
}

pub struct StateServer {
    /// position
    p: Vector3d,
    /// velocity
    v: Vector3d,
    /// rotation matrix
    rot: Matrix3d,
    /// gyro bias  
    bg: Vector3d,
    /// acc bias
    ba: Vector3d,
    /// transformation between IMU and left camera frame
    r_imu_cam0: Matrix3d,
    t_cam0_imu: Vector3d,
    // noise matrix
    q_mat: Matrixd,
    // covariance matrix
    state_cov: Matrixd,
    gravity: Vector3d,
    window_size: usize,
    last_time: Option<f64>,
    is_initialized: bool,
    camera_states: HashMap<usize, CameraState>,
}

impl StateServer {
    pub fn new() -> Self {
        let config = CONFIG.get().unwrap();

        let gravity = Vector3d::new(0., 0., -config.gravity);
        let window_size = config.window_size;

        // state is R, v, p, bg, ba, transformation of imu wrt cam
        // so len is 3 x 7 = 21
        let state_cov = Matrixd::zeros(STATE_LEN, STATE_LEN);
        let q_mat = Matrixd::zeros(STATE_LEN, STATE_LEN);

        Self {
            p: Vector3d::zeros(),
            v: Vector3d::zeros(),
            rot: Matrix3d::identity(),
            bg: Vector3d::zeros(),
            ba: Vector3d::zeros(),
            r_imu_cam0: Matrix3d::identity(),
            t_cam0_imu: Vector3d::zeros(),
            state_cov,
            q_mat,
            gravity,
            window_size,
            last_time: None,
            is_initialized: false,
            camera_states: HashMap::new(),
        }
    }

    pub fn initialize_q(&mut self) {
        let config = CONFIG.get().unwrap();

        let set_diagonal = |x: &mut Matrixd, start: usize, len: usize, value: f64| {
            for i in start..(start + len) {
                x[(i, i)] = value.powi(2);
            }
        };

        set_diagonal(&mut self.q_mat, 0, 3, config.gyro_var);
        set_diagonal(&mut self.q_mat, 3, 3, config.acc_var);
        set_diagonal(&mut self.q_mat, 9, 3, config.bias_gyro_var);
        set_diagonal(&mut self.q_mat, 12, 3, config.bias_acc_var);
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
        // in the order (w, i, j, k)
        let q = na::Quaternion::new(w, xyz[(0, 0)], xyz[(1, 0)], xyz[(2, 0)]);
        let q = na::UnitQuaternion::from_quaternion(q);
        self.rot = q.to_rotation_matrix().into();
    }

    /// ref stereo msckf processModel
    pub fn predict(&mut self, time: f64, gyro: Vector3d, acc: Vector3d) {
        // initialize the orientation and Q
        if !self.is_initialized {
            self.initialize_orientation(acc);
            self.initialize_q();
            self.is_initialized = true;
        }

        // update dt
        let dt = if let Some(last_time) = self.last_time {
            time - last_time
        } else {
            0.
        };
        self.last_time = Some(time);
        if dt <= 0. {
            return;
        }

        // remove the bias
        let gyro = gyro - self.bg;
        let acc = acc - self.ba;

        // update imu position
        let new_p = self.p
            + self.v * dt
            + 0.5 * (self.rot * acc) * dt.powi(2)
            + 0.5 * self.gravity * dt.powi(2);

        // update imu velocity
        let new_v = self.v + self.rot * acc * dt + self.gravity * dt;

        // update imu orientation
        let new_rot = self.rot * so3_exp(&(gyro * dt));

        self.rot = new_rot;
        self.v = new_v;
        self.p = new_p;

        let f_mat = self.construct_f_mat(dt, &acc, &gyro);
        let imu_state_cov = self.state_cov.fixed_slice::<21, 21>(0, 0).clone();
        self.state_cov.fixed_slice_mut::<21, 21>(0, 0).copy_from(
            &(f_mat * imu_state_cov * f_mat.transpose()
                + f_mat * self.q_mat * f_mat.transpose() * dt),
        );
        if self.camera_states.len() > 0 {
            let cov_slice = f_mat
                * self
                    .state_cov
                    .slice((0, 21), (21, self.state_cov.ncols() - 21));
            self.state_cov.slice_mut((0, 21), (21, self.state_cov.ncols() - 21)).copy_from(&cov_slice); 

            let cov_slice = self
                    .state_cov
                    .slice((21, 0), (self.state_cov.nrows() - 21, 21)) * f_mat.transpose();
            self.state_cov.slice_mut((21, 0), (self.state_cov.nrows() - 21, 21)).copy_from(&cov_slice);  
        }
    }

    pub fn construct_f_mat(&self, dt: f64, acc: &Vector3d, gyro: &Vector3d) -> Matrixd {
        // Compute discrete transition and noise covariance matrix
        let mut f_mat: Matrixd = Matrixd::identity(STATE_LEN, STATE_LEN);
        let i3 = Matrix3d::identity();

        // theta wrt theta
        f_mat
            .fixed_slice_mut::<3, 3>(0, 0)
            .copy_from(&(so3_exp(&(-gyro * dt))));
        // theta vs bg
        f_mat.fixed_slice_mut::<3, 3>(0, 9).copy_from(&(-i3 * dt));

        // v wrt theta
        f_mat
            .fixed_slice_mut::<3, 3>(3, 0)
            .copy_from(&(-self.rot * skew(acc) * dt));
        // v wrt ba
        f_mat
            .fixed_slice_mut::<3, 3>(3, 12)
            .copy_from(&(-self.rot * dt));

        // p wrt v
        f_mat.fixed_slice_mut::<3, 3>(6, 3).copy_from(&(i3 * dt));

        f_mat
    }
}
