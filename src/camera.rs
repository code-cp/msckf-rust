use std::fmt::Debug;
use nalgebra as na; 

use crate::my_types::*;

#[derive(Debug)]
pub struct Camera {
    pub kind: CameraKind,
    pub model: Box<dyn CameraModel>,
    pub image_shape: ImageShape, 
}

#[derive(Debug)]
pub enum CameraKind {
    Pinhole,
}

pub trait CameraModel: Debug {
    fn back_project(&self, pixel: Vector2d) -> Option<Vector3d>;

    fn project_with_derivative(
        &self,
        ray: Vector3d,
        compute_derivative: bool,
    ) -> (Option<Vector2d>, Option<Matrix23d>);

    fn project(&self, ray: Vector3d) -> Option<Vector2d> {
        self.project_with_derivative(ray, false).0
    }
}

#[derive(Debug, Clone)]
pub struct CameraPose {
    pub id: usize,
    /// Take a vector from the camera frame to world frame
    pub orientation: Matrix3d,
    pub position: Vector3d,
}

impl CameraPose {
    pub fn inverse(&self) -> Self {
        let orientation = self.orientation.transpose(); 
        let position = -orientation * self.position; 

        Self {
            id: self.id, 
            orientation, 
            position, 
        }
    }
}

#[derive(Debug, Clone)]
pub struct CameraState {
    pub id: usize,
    /// Take a vector from the camera frame to world frame
    pub orientation: Matrix3d,
    pub position: Vector3d,
    pub cam1_trans_cam0: CameraPose, 
}

impl CameraState {
    pub fn new(id: usize, orientation: Matrix3d, position: Vector3d, r_cam0_cam1: Matrix3d, t_cam0_cam1: Vector3d) -> Self {
        let cam1_trans_cam0 = CameraPose {
            id, 
            orientation: r_cam0_cam1, 
            position: t_cam0_cam1, 
        }; 
    
        Self {
            id, 
            orientation, 
            position, 
            cam1_trans_cam0, 
        }
    }

    pub fn convert_to_stereo_poses_vec(&self) -> Vec<CameraPose> {
        let right_orientation =  self.orientation * self.cam1_trans_cam0.orientation.transpose();
        let right_position = -self.orientation * self.cam1_trans_cam0.orientation.transpose() * self.cam1_trans_cam0.position + self.position; 

        let left_pose = CameraPose {
            id: self.id, 
            orientation: self.orientation, 
            position: self.position, 
        }; 
        let right_pose = CameraPose {
            id: self.id, 
            orientation: right_orientation, 
            position: right_position, 
        }; 
        vec![left_pose, right_pose]
    }
}
