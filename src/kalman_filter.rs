use nalgebra as na;

use crate::my_types::*; 
use crate::config::*;
use crate::math::*;

pub struct ESKF {
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
    state_len: usize, 
    // noise matrix 
    q_mat: Matrixd, 
    // covariance matrix  
    state_cov: Matrixd, 
    gravity: Vector3d, 
    window_size: usize, 
    last_time: Option<f64>, 
    is_initialized: bool, 
}

impl ESKF {
    pub fn new() -> Self {
        let config = CONFIG.get().unwrap(); 

        let gravity = Vector3d::new(0., 0., -config.gravity);
        let window_size = config.window_size;

        // state is p, v, R, bg, ba, g 
        // so len is 3 x 6 = 18 
        let state_len = 18; 
        let state_cov = Matrixd::zeros(state_len, state_len); 
        let q_mat = Matrixd::zeros(state_len, state_len); 

        let set_diagonal_from_noise = | x: &mut Matrixd, start, len: usize, std_value: f64| {
            for i in start..(start + len) {
                x[(i, i)] = std_value.powi(2); 
            }
        };

        Self {
            p: Vector3d::zeros(), 
            v: Vector3d::zeros(),
            rot: Matrix3d::identity(),
            bg: Vector3d::zeros(), 
            ba: Vector3d::zeros(),
            state_cov, 
            q_mat, 
            state_len, 
            gravity, 
            window_size, 
            last_time: None, 
            is_initialized: false, 
        }
    }

    pub fn initialize_q(&mut self) {
        let config = CONFIG.get().unwrap();  

        let set_diagonal = |x: &mut Matrixd, start: usize, len: usize, value: f64| {
            for i in start..(start + len) {
              x[(i, i)] = value.powi(2);
            }
          };

        set_diagonal(&mut self.q_mat, 9, 3, config.acc_var); 
        set_diagonal(&mut self.q_mat, 12, 3, config.gyro_var); 
        set_diagonal(&mut self.q_mat, 15, 3, config.bias_gyro_var);
        set_diagonal(&mut self.q_mat, 18, 3, config.bias_acc_var); 
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
    pub fn predict(
        &mut self, 
        time: f64, 
        gyro: Vector3d, 
        acc: Vector3d, 
    ) {
        // initialize the orientation and Q
        if !self.is_initialized {
            self.initialize_orientation(acc);
            self.initialize_q();
            self.is_initialized = true; 
        }

        // update dt 
        let dt = if let Some(last_time) = self.last_time {time - last_time} else { 0. }; 
        self.last_time = Some(time); 
        if dt <= 0. { return; }

        // remove the bias 
        let gyro = gyro - self.bg; 
        let acc = acc - self.ba; 

        // update imu position 
        let new_p = self.p + self.v * dt + 0.5 * (self.rot * acc) * dt.powi(2) + 0.5 * self.gravity * dt.powi(2);  

        // update imu velocity 
        let new_v = self.v + self.rot * acc * dt + self.gravity * dt; 

        // update imu orientation
        let new_rot = self.rot * so3_exp(&(gyro * dt));

        self.rot = new_rot; 
        self.v = new_v; 
        self.p = new_p; 

        let f_mat = self.construct_f_mat(dt, &acc, &gyro);
        self.state_cov = f_mat * self.state_cov * f_mat.transpose() + self.q_mat; 
    }

    pub fn construct_f_mat(&self, dt: f64, acc: &Vector3d, gyro: &Vector3d) -> Matrixd {
        // Compute discrete transition and noise covariance matrix
        let mut f_mat: Matrixd = Matrixd::identity(self.state_len, self.state_len); 
        let i3 = Matrix3d::identity(); 

        // p wrt v 
        f_mat.fixed_slice_mut::<3, 3>(0, 3).copy_from(&(i3 * dt)); 
        // v wrt theta 
        f_mat.fixed_slice_mut::<3, 3>(3, 6).copy_from(&(-self.rot * skew(acc) * dt)); 
        // v wrt ba 
        f_mat.fixed_slice_mut::<3, 3>(3, 12).copy_from(&(-self.rot * dt));
        // v wrt g 
        f_mat.fixed_slice_mut::<3, 3>(3, 15).copy_from(&(i3 * dt));
        // theta wrt theta 
        f_mat.fixed_slice_mut::<3, 3>(6, 6).copy_from(&(so3_exp(&(-gyro*dt))));
        // theta vs bg 
        f_mat.fixed_slice_mut::<3, 3>(6, 9).copy_from(&(-i3*dt)); 

        f_mat
    }

}