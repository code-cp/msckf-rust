use std::os::macos::raw::stat;

use nalgebra as na; 

use crate::my_types::*; 
use crate::config::*;

pub struct KalmanFilter {
    state: Vectord, 
    state_len: usize, 
    state_covariance: Matrixd, 

    gravity: Vector3d, 
}

impl KalmanFilter {
    pub fn new() -> Self {
        let config = CONFIG.get().unwrap(); 

        let gravity = Vector3d::new(0., 0., -config.gravity);

        let state_len = 6;
        let state = na::DVector::zeros(state_len);
        let mut state_covariance = na::DMatrix::zeros(state_len, state_len); 

        Self {
            state, 
            state_len,
            state_covariance, 

            gravity, 
        }
    }

    pub fn predict(
        &mut self, 
        time: f64, 
        gyro: Vector3d, 
        acc: Vector3d, 
    ) {

    }
}