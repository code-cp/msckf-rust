use nalgebra as na;
use nalgebra_lapack::SVD as LapackSVD;
use nalgebra_lapack::QR as LapackQR;
use rand::seq::SliceRandom;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::{BTreeMap, HashMap};

use crate::camera::CameraState;
use crate::config::*;
use crate::feature::{triangulate, Track};
use crate::math::*;
use crate::my_types::*;
use crate::stationary::*;

static STATE_LEN: usize = 21;

pub struct Extrinsics {
    pub trans_imu_cam0: Matrix4d,
    pub trans_cam0_cam1: Matrix4d,
}

#[derive(Debug)]
enum InitializationMethod {
    FromImuData, 
    FromGroundtruth, 
}

#[derive(Debug)]
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
    trans_imu_cam0: Matrix4d, 
    r_imu_cam0: Matrix3d,
    t_cam0_imu: Vector3d,
    /// Takes a vector from the cam0 frame to the cam1 frame
    trans_cam0_cam1: Matrix4d,
    r_cam0_cam1: Matrix3d,
    t_cam0_cam1: Vector3d,
    /// noise matrix
    q_mat: Matrixd,
    /// covariance matrix
    state_cov: Matrixd,
    gravity: Vector3d,
    window_size: usize,
    last_time: Option<f64>,
    pub is_initialized: bool,
    camera_states: BTreeMap<usize, CameraState>,
    pub state_id: usize,
    pub feature_map: HashMap<usize, Vector3d>,
    pub stationary: Stationary,
    initialization_method: InitializationMethod, 
    first_pose_gt: Matrix4d, 
    pub current_pose_gt: Option<Matrix4d>, 
}

fn set_diagonal(x: &mut Matrixd, start: usize, len: usize, value: f64) {
    for i in start..(start + len) {
        x[(i, i)] = value.powi(2);
    }
}

impl StateServer {
    pub fn new(extrinsics: &Extrinsics, first_pose_gt: &Matrix4d) -> Self {
        let config = CONFIG.get().unwrap();

        let gravity = Vector3d::new(0., 0., -config.gravity);
        let window_size = config.window_size;

        // state is R, v, p, bg, ba, transformation of imu wrt cam
        // so len is 3 x 7 = 21
        let state_cov = Matrixd::zeros(STATE_LEN, STATE_LEN);
        let q_mat = Matrixd::zeros(STATE_LEN, STATE_LEN);

        let trans_imu_cam0 = extrinsics.trans_imu_cam0;
        let trans_cam0_cam1 = extrinsics.trans_cam0_cam1;

        let r_imu_cam0 = trans_imu_cam0.fixed_view::<3, 3>(0, 0);
        let trans_cam0_imu = trans_imu_cam0.try_inverse().unwrap();
        let t_cam0_imu = trans_cam0_imu.fixed_view::<3, 1>(0, 3);

        let r_cam0_cam1 = trans_cam0_cam1.fixed_view::<3, 3>(0, 0);
        let t_cam0_cam1 = trans_cam0_cam1.fixed_view::<3, 1>(0, 3);

        let initialization_method = InitializationMethod::FromGroundtruth; 

        Self {
            p: Vector3d::zeros(),
            v: Vector3d::zeros(),
            rot: Matrix3d::identity(),
            bg: Vector3d::zeros(),
            ba: Vector3d::zeros(),
            trans_imu_cam0, 
            r_imu_cam0: r_imu_cam0.into(),
            t_cam0_imu: t_cam0_imu.into(),
            r_cam0_cam1: r_cam0_cam1.into(),
            t_cam0_cam1: t_cam0_cam1.into(),
            trans_cam0_cam1,
            state_cov,
            q_mat,
            gravity,
            window_size,
            last_time: None,
            is_initialized: false,
            camera_states: BTreeMap::new(),
            state_id: 0,
            feature_map: HashMap::new(),
            stationary: Stationary::new(),
            initialization_method, 
            first_pose_gt: first_pose_gt.to_owned(), 
            current_pose_gt: None, 
        }
    }

    /// obtain the camera pose in world frame for visualization
    pub fn get_camera_pose(&self) -> Matrix4d {
        let w_r_cam0 = self.rot * self.r_imu_cam0.transpose();
        let w_t_cam0 = self.rot * self.t_cam0_imu + self.p;
        let mut se3 = Matrix4d::zeros();
        se3.fixed_view_mut::<3, 3>(0, 0).copy_from(&w_r_cam0);
        se3.fixed_view_mut::<3, 1>(0, 3).copy_from(&w_t_cam0);
        se3
    }

    /// The initial covariance of orientation and position can be
    /// set to 0. But for velocity, bias and extrinsic parameters, 
    /// there should be nontrivial uncertainty.
    pub fn initialize_state_cov(&mut self) {
        set_diagonal(&mut self.state_cov, 3, 3, 0.25);
        set_diagonal(&mut self.state_cov, 9, 3, 0.01);
        set_diagonal(&mut self.state_cov, 12, 3, 0.01);
        set_diagonal(&mut self.state_cov, 15, 3, 3.0462e-4);
        set_diagonal(&mut self.state_cov, 18, 3, 2.5e-5);
    }

    pub fn initialize_q(&mut self) {
        let config = CONFIG.get().unwrap();

        set_diagonal(&mut self.q_mat, 0, 3, config.gyro_std);
        set_diagonal(&mut self.q_mat, 3, 3, config.acc_std);
        set_diagonal(&mut self.q_mat, 9, 3, config.bias_gyro_std);
        set_diagonal(&mut self.q_mat, 12, 3, config.bias_acc_std);
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

        // // quaternion is represented by wxyz
        // // Hamilton convention
        // // from world to IMU
        // // in the order (w, i, j, k)
        // let q = na::Quaternion::new(w, xyz[(0, 0)], xyz[(1, 0)], xyz[(2, 0)]);
        // let q = na::UnitQuaternion::from_quaternion(q);
        // // from IMU to world
        // self.rot = q.to_rotation_matrix().transpose().into();

        let q = Vector4d::new(w, xyz[(0, 0)], xyz[(1, 0)], xyz[(2, 0)]);
        self.rot = to_rotation_matrix(q).transpose();
    }

    /// ref stereo msckf processModel
    pub fn predict(&mut self, time: f64, gyro: Vector3d, acc: Vector3d) {
        // initialize the orientation and Q
        if !self.is_initialized {
            match self.initialization_method {
                InitializationMethod::FromGroundtruth => {
                    self.p = self.first_pose_gt.fixed_view::<3,1>(0,3).into(); 
                    self.rot = self.first_pose_gt.fixed_view::<3,3>(0,0).into();
                }
                InitializationMethod::FromImuData => {
                    self.initialize_orientation(acc);
                }
            }
            
            self.initialize_state_cov(); 
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
        let imu_state_cov = self.state_cov.view((0, 0), (STATE_LEN, STATE_LEN)).clone();
        let temp_block = f_mat.clone() * imu_state_cov * f_mat.clone().transpose();
        self.state_cov.view_mut((0, 0), (21, 21)).copy_from(
            &(temp_block + f_mat.clone() * self.q_mat.clone() * f_mat.clone().transpose() * dt),
        );

        if self.camera_states.len() > 0 {
            let cov_slice = f_mat.clone()
                * self
                    .state_cov
                    .view((0, STATE_LEN), (STATE_LEN, self.state_cov.ncols() - STATE_LEN));
            self.state_cov
                .view_mut((0, STATE_LEN), (STATE_LEN, self.state_cov.ncols() - STATE_LEN))
                .copy_from(&cov_slice);

            let cov_slice = self
                .state_cov
                .view((STATE_LEN, 0), (self.state_cov.nrows() - STATE_LEN, STATE_LEN))
                * f_mat.transpose();
            self.state_cov
                .view_mut((STATE_LEN, 0), (self.state_cov.nrows() - STATE_LEN, STATE_LEN))
                .copy_from(&cov_slice);
        }
    }

    pub fn construct_f_mat(&self, dt: f64, acc: &Vector3d, gyro: &Vector3d) -> Matrixd {
        // Compute discrete transition and noise covariance matrix
        let mut f_mat: Matrixd = Matrixd::identity(STATE_LEN, STATE_LEN);
        let i3 = Matrix3d::identity();

        // theta wrt theta
        f_mat
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(so3_exp(&(-gyro * dt))));
        // theta vs bg
        f_mat.fixed_view_mut::<3, 3>(0, 9).copy_from(&(-i3 * dt));

        // v wrt theta
        f_mat
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(-self.rot * skew(acc) * dt));
        // v wrt ba
        f_mat
            .fixed_view_mut::<3, 3>(3, 12)
            .copy_from(&(-self.rot * dt));

        // p wrt v
        f_mat.fixed_view_mut::<3, 3>(6, 3).copy_from(&(i3 * dt));

        f_mat
    }

    pub fn augment_state(&mut self) {
        let c_r_i = self.r_imu_cam0.clone();
        let i_p_c = self.t_cam0_imu.clone();

        let w_r_i = self.rot.clone();
        let w_p_i = self.p.clone();
        let w_r_c = w_r_i * c_r_i.transpose();
        let w_p_c = w_p_i + w_r_i * i_p_c;

        let camera_state = CameraState::new(
            self.state_id,
            w_r_c,
            w_p_c,
            self.r_cam0_cam1,
            self.t_cam0_cam1,
        );

        self.camera_states.insert(self.state_id, camera_state);

        let dcampose_dimupose =
            self.get_cam_wrt_imu_se3_jacobian(&c_r_i, &i_p_c, &w_r_c.transpose());
        let mut jacobian = Matrixd::zeros(6, STATE_LEN);
        jacobian
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&dcampose_dimupose.fixed_view::<3, 3>(0, 0));
        jacobian
            .fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&dcampose_dimupose.fixed_view::<3, 3>(0, 3));
        jacobian
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&dcampose_dimupose.fixed_view::<3, 3>(3, 0));
        jacobian
            .fixed_view_mut::<3, 3>(3, 6)
            .copy_from(&dcampose_dimupose.fixed_view::<3, 3>(3, 3));
        jacobian
            .fixed_view_mut::<3, 3>(0, 15)
            .copy_from(&Matrix3d::identity());
        jacobian
            .fixed_view_mut::<3, 3>(3, 18)
            .copy_from(&Matrix3d::identity());

        let old_size = self.state_cov.nrows();
        let mut state_cov = Matrixd::zeros(old_size + 6, old_size + 6);
        state_cov
            .view_mut((0, 0), (old_size, old_size))
            .copy_from(&self.state_cov);
        let block = jacobian.clone() * self.state_cov.view((0, 0), (21, old_size));
        state_cov
            .view_mut((old_size, 0), (6, old_size))
            .copy_from(&block);
        state_cov
            .view_mut((0, old_size), (old_size, 6))
            .copy_from(&(block.transpose()));
        state_cov.view_mut((old_size, old_size), (6, 6)).copy_from(
            &(jacobian.clone() * self.state_cov.view((0, 0), (STATE_LEN, STATE_LEN)) * jacobian.transpose()),
        );
        self.state_cov = state_cov;
    }

    pub fn prune_state(&mut self) {
        if self.camera_states.len() <= self.window_size {
            return;
        }
        // remove the oldest camera
        self.camera_states.pop_first();

        let cam_state_start = STATE_LEN;
        let cam_state_end = cam_state_start + 6;
        let new_size = STATE_LEN + self.window_size * 6;
        let mut new_state_cov = Matrixd::zeros(new_size, new_size);
        new_state_cov
            .view_mut((0, 0), (STATE_LEN, STATE_LEN))
            .copy_from(&self.state_cov.view((0, 0), (STATE_LEN, STATE_LEN)));
        new_state_cov
            .view_mut((0, STATE_LEN), (STATE_LEN, self.window_size * 6))
            .copy_from(
                &self
                    .state_cov
                    .view((0, cam_state_end), (STATE_LEN, self.window_size * 6)),
            );
        new_state_cov
            .view_mut((STATE_LEN, 0), (self.window_size * 6, STATE_LEN))
            .copy_from(
                &self
                    .state_cov
                    .view((cam_state_end, 0), (self.window_size * 6, STATE_LEN)),
            );
        new_state_cov
            .view_mut(
                (STATE_LEN, STATE_LEN),
                (self.window_size * 6, self.window_size * 6),
            )
            .copy_from(&self.state_cov.view(
                (cam_state_end, cam_state_end),
                (self.window_size * 6, self.window_size * 6),
            ));
        self.state_cov = new_state_cov;
    }

    fn get_cam_wrt_imu_se3_jacobian(
        &self,
        c_r_i: &Matrix3d,
        i_p_c: &Vector3d,
        c_r_w: &Matrix3d,
    ) -> na::Matrix6<f64> {
        let mut p_cxi_p_ixi = na::Matrix6::<f64>::zeros();
        p_cxi_p_ixi
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(-1.0 * c_r_i * skew(i_p_c)));
        p_cxi_p_ixi.fixed_view_mut::<3, 3>(3, 0).copy_from(c_r_i);
        p_cxi_p_ixi.fixed_view_mut::<3, 3>(0, 3).copy_from(c_r_w);
        p_cxi_p_ixi
    }

    pub fn get_feature_map_for_visualization(&self) -> Vec<[f32; 3]> {
        self.feature_map
            .values()
            .copied()
            .map(|value| [value[0] as f32, value[1] as f32, value[2] as f32])
            .collect()
    }

    fn get_camera_indices(&self) -> Vec<usize> {
        let camera_frame_indices: Vec<usize> =
            self.camera_states.keys().copied().collect::<Vec<usize>>();
        camera_frame_indices
    }

    /// Assume velocity measurement is zero 
    /// alternatively, can assume angular velocity and linear acc are zero, eg 
    /// ref <https://docs.openvins.com/update-zerovelocity.html>
    pub fn update_zero_velocity(&mut self) {
        let mut jacobian_x = Matrixd::zeros(3, STATE_LEN + 6 * self.camera_states.len());
        
        // residual is z - predicted value 
        let mut residual = Vectord::zeros(0);
        residual.resize_vertically_mut(3, 0.);
        residual.fixed_view_mut::<3, 1>(0, 0).copy_from(&(-self.v));

        jacobian_x.fixed_view_mut::<3, 3>(0, 3).copy_from(&Matrix3d::identity());

        let noise: f64 = (0.001_f64).powi(2);
        let _ = self.kf_update(&jacobian_x.into(), &residual.into(), noise); 
    }

    pub fn update_pose(&mut self, pose: &Matrix4d) {
        // pose is imu to world 
        let camera_pose = pose * self.trans_imu_cam0.try_inverse().unwrap(); 
        self.current_pose_gt = Some(camera_pose.to_owned()); 

        let mut jacobian_x = Matrixd::zeros(6, STATE_LEN + 6 * self.camera_states.len());
        
        // residual is z - expected value 
        let mut residual = Vectord::zeros(0);
        residual.resize_vertically_mut(6, 0.);

        let position_measurement = pose.fixed_view::<3,1>(0,3);  
        residual.fixed_view_mut::<3, 1>(0, 0).copy_from(&(position_measurement - self.p));

        let orientation_measurement = pose.fixed_view::<3,3>(0,0); 
        let orientation_residual = self.rot.transpose() * orientation_measurement; 
        let orientation_residual = rotation_matrix_to_angle_axis(&orientation_residual); 
        residual.fixed_view_mut::<3,1>(3, 0).copy_from(&orientation_residual); 

        jacobian_x.fixed_view_mut::<3, 3>(0, 6).copy_from(&Matrix3d::identity());
        jacobian_x.fixed_view_mut::<3, 3>(3, 0).copy_from(&Matrix3d::identity());

        let noise = 1e-5; 
        let _ = self.kf_update(&jacobian_x.into(), &residual.into(), noise);
    }

    pub fn update_feature(&mut self, tracks: &[Track]) {
        let feature_update_number = 50;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);

        let camera_frame_indices = self.get_camera_indices();

        let mut jacobian_x_all = None; 
        let mut residual_all = None; 

        'track: for track in tracks.choose_multiple(&mut rng, feature_update_number) {
            let mut normalized_coordinates = Vec::<[Vector2d; 2]>::new();
            let mut camera_poses = Vec::<&CameraState>::new();
            for point in &track.points {
                match camera_frame_indices.binary_search(&point.frame_number) {
                    Ok(frame_index) => {
                        normalized_coordinates.push(point.normalized_coordinates);
                        let camera_idx = camera_frame_indices[frame_index];
                        camera_poses.push(&self.camera_states.get(&camera_idx).unwrap());
                    }
                    Err(_) => {
                        continue;
                    }
                }
            }

            if normalized_coordinates.len() == 0 {
                continue;
            }

            if let Some(position) = triangulate(&normalized_coordinates, &camera_poses) {
                // update feature map
                self.feature_map
                    .entry(track.id.0)
                    .and_modify(|value| *value = position)
                    .or_insert(position);

                // update step
                let jacobian_row_size = 4 * camera_poses.len();
                let mut jacobian_x =
                    Matrixd::zeros(jacobian_row_size, STATE_LEN + 6 * self.camera_states.len());
                let mut jacobian_f = Matrixd::zeros(jacobian_row_size, 3);
                let mut residual = Vectord::zeros(jacobian_row_size);
                for ((observation_index, normalized_coordinate), pose) in normalized_coordinates
                    .iter()
                    .enumerate()
                    .zip(camera_poses.iter())
                {
                    // Convert the feature position from the world frame to camera frame
                    let stereo_poses = pose.convert_to_stereo_poses_vec();
                    let w_trans_left_cam = &stereo_poses[0];
                    let left_cam_trans_w = w_trans_left_cam.inverse();
                    let w_trans_right_cam = &stereo_poses[1];
                    let right_cam_trans_w = w_trans_right_cam.inverse();

                    let p_c0 = left_cam_trans_w.orientation * position + left_cam_trans_w.position;
                    let p_c1 =
                        right_cam_trans_w.orientation * position + right_cam_trans_w.position;

                    // Check the triangulated point is in front of all cameras
                    if p_c0[2] < 0. || p_c1[2] < 0. {
                        continue 'track;
                    }

                    // Compute the Jacobians of the reprojection error wrt the state
                    let mut temp_mat = Matrixd::zeros(3, 4);
                    temp_mat
                        .view_mut((0, 0), (3, 3))
                        .copy_from(&Matrix3d::identity());

                    let p_c0_jacobi = p_c0;
                    let mut dz_dpc0 = Matrixd::zeros(4, 3);
                    dz_dpc0[(0, 0)] = 1. / p_c0[2];
                    dz_dpc0[(0, 2)] = -p_c0[0] / (p_c0[2].powi(2));
                    dz_dpc0[(1, 1)] = 1. / p_c0[2];
                    dz_dpc0[(1, 2)] = -p_c0[1] / (p_c0[2].powi(2));
                    let uline_l0 =
                        Vector4d::new(p_c0_jacobi[0], p_c0_jacobi[1], p_c0_jacobi[2], 1.);
                    let dpc0_dxc = -temp_mat.clone() * odot_operator(&uline_l0);

                    let mut dz_dpc1 = Matrixd::zeros(4, 3);
                    dz_dpc1[(2, 0)] = 1. / p_c1[2];
                    dz_dpc1[(2, 2)] = -p_c1[0] / (p_c1[2].powi(2));
                    dz_dpc1[(3, 1)] = 1. / p_c1[2];
                    dz_dpc1[(3, 2)] = -p_c1[1] / (p_c1[2].powi(2));
                    let dpc1_dxc =
                        -temp_mat.clone() * self.trans_cam0_cam1 * odot_operator(&uline_l0);

                    let dpc0_dpg = left_cam_trans_w.orientation;
                    let dpc1_dpg = right_cam_trans_w.orientation;

                    //shape (4, 6)
                    let jacobian_x_feature =
                        dz_dpc0.clone() * dpc0_dxc + dz_dpc1.clone() * dpc1_dxc;
                    //shape (4, 3)
                    let jacobian_f_feature = dz_dpc0 * dpc0_dpg + dz_dpc1 * dpc1_dpg;

                    // Compute the residual
                    let mut residual_feature = Vector4d::zeros();
                    residual_feature[0] = normalized_coordinate[0][0] - p_c0[0] / p_c0[2];
                    residual_feature[1] = normalized_coordinate[0][1] - p_c0[1] / p_c0[2];
                    residual_feature[2] = normalized_coordinate[1][0] - p_c1[0] / p_c1[2];
                    residual_feature[3] = normalized_coordinate[1][1] - p_c1[1] / p_c1[2];

                    let camera_index = camera_frame_indices
                        .iter()
                        .position(|&v| v == pose.id)
                        .unwrap();
                    jacobian_x
                        .view_mut(
                            (observation_index * 4, STATE_LEN + 6 * camera_index),
                            (4, 6),
                        )
                        .copy_from(&jacobian_x_feature);
                    jacobian_f
                        .view_mut((observation_index * 4, 0), (4, 3))
                        .copy_from(&jacobian_f_feature);
                    residual
                        .view_mut((observation_index * 4, 0), (4, 1))
                        .copy_from(&residual_feature);
                }

                // Project the residual and Jacobians onto the nullspace
                // let svd = na::linalg::SVD::new(jacobian_f.clone(), true, true);
                let svd = LapackSVD::new(jacobian_f);
                if svd.is_none() {
                    continue;
                }
                let svd = svd.unwrap();

                // U matrix
                let u_mat = svd.u;
                let a_mat = u_mat.view((0, 3), (u_mat.nrows(), u_mat.ncols() - 3));
                let jacobian_x = a_mat.transpose() * jacobian_x;
                let residual = a_mat.transpose() * residual;
            
                if let (Some(jacobian_x_all), Some(residual_all)) = (jacobian_x_all.as_mut(), residual_all.as_mut()) {
                    *jacobian_x_all = hstack_mat(&jacobian_x_all, &jacobian_x);
                    *residual_all = hstack_vec(&residual_all, &residual);
                } else {
                    jacobian_x_all = Some(jacobian_x); 
                    residual_all = Some(residual); 
                } 
            }

        }

        if let (Some(jacobian_x), Some(residual)) = (jacobian_x_all, residual_all) {
            let observation_noise: f64 = (0.035_f64).powi(2);
            let delta_x = self.kf_update(&jacobian_x.into(), &residual.into(), observation_noise); 
        } else {
            return;
        }
    }

    fn kf_update(&mut self, jacobian_x: &Matrixd, residual: &Vectord, noise: f64) -> Vectord {
        // QR decomposition
        let mut jacobian_x = jacobian_x.to_owned(); 
        let mut residual = residual.to_owned();    

        if jacobian_x.nrows() > jacobian_x.ncols() {
            let qr = LapackQR::new(jacobian_x);
            jacobian_x = qr.r(); 
            let q: Matrixd = qr.q(); 
            residual = q.transpose() * residual; 
        }

        // perform update step
        let p_mat = self.state_cov.clone();
        let jacobian_x_transpose = jacobian_x.transpose();
        let nrows = jacobian_x.nrows();
        let s_mat = jacobian_x.clone() * p_mat.clone() * jacobian_x_transpose
            + noise * Matrixd::identity(nrows, nrows);
        
        // let tolerance = 1e-3;
        // let k_mat = na::linalg::SVD::new(s_mat, true, true)
        //     .solve(&(jacobian_x.clone() * p_mat), tolerance)
        //     .unwrap()
        //     .transpose();
        let k_mat = p_mat * jacobian_x.clone().transpose() * s_mat.try_inverse().unwrap(); 

        let delta_x = k_mat.clone() * residual;

        // update the imu state
        self.rot = self.rot * so3_exp(&Vector3d::new(delta_x[0], delta_x[1], delta_x[2]));
        self.v += Vector3d::new(delta_x[3], delta_x[4], delta_x[5]);
        self.p += Vector3d::new(delta_x[6], delta_x[7], delta_x[8]);
        self.bg += Vector3d::new(delta_x[9], delta_x[10], delta_x[11]);
        self.ba += Vector3d::new(delta_x[12], delta_x[13], delta_x[14]);

        // update the extrinsics
        let delta_i_trans_c = se3_exp(&Vector6d::new(
            delta_x[15],
            delta_x[16],
            delta_x[17],
            delta_x[18],
            delta_x[19],
            delta_x[20],
        ));
        // need to update P first using old R
        let mut i_r_c = self.r_imu_cam0.transpose();
        let i_p_c = self.t_cam0_imu;
        self.t_cam0_imu = i_r_c
            * Vector3d::new(
                delta_i_trans_c[(0, 3)],
                delta_i_trans_c[(1, 3)],
                delta_i_trans_c[(2, 3)],
            )
            + i_p_c;

        // update R via right perturbation 
        i_r_c = i_r_c * delta_i_trans_c.fixed_view::<3, 3>(0, 0);
        self.r_imu_cam0 = i_r_c.transpose();

        // Update the camera states
        let camera_frame_indices = self.get_camera_indices();
        for (index, camera_index) in camera_frame_indices.iter().enumerate() {
            let delta_camera = na::Vector6::new(
                delta_x[STATE_LEN + index * 6],
                delta_x[STATE_LEN + index * 6 + 1],
                delta_x[STATE_LEN + index * 6 + 2],
                delta_x[STATE_LEN + index * 6 + 3],
                delta_x[STATE_LEN + index * 6 + 4],
                delta_x[STATE_LEN + index * 6 + 5],
            );
            let delta_w_trans_cam = se3_exp(&delta_camera);
            let camera_state = self.camera_states.get(camera_index).unwrap().clone();
            let orientation = camera_state.orientation;
            let position = camera_state.position;
            let camera_state = self.camera_states.get_mut(camera_index).unwrap();
            camera_state.position = orientation
                * Vector3d::new(
                    delta_w_trans_cam[(0, 3)],
                    delta_w_trans_cam[(1, 3)],
                    delta_w_trans_cam[(2, 3)],
                )
                + position;
            camera_state.orientation =
                orientation * delta_w_trans_cam.fixed_view::<3, 3>(0, 0);
        }

        // Update state covariance
        let i_kh_mat = Matrixd::identity(k_mat.nrows(), k_mat.nrows()) - k_mat * jacobian_x;
        let state_cov = self.state_cov.clone();
        let state_cov = i_kh_mat * state_cov;

        // Fix the covariance to be symmetric
        self.state_cov = (state_cov.clone() + state_cov.transpose()) / 2.0; 

        return delta_x;
    } 
}
