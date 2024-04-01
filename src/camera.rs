use std::fmt::Debug;

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

#[derive(Debug)]
pub struct CameraState {
    pub id: usize,
    /// Take a vector from the camera frame to world frame
    pub left_orientation: Matrix3d,
    pub left_position: Vector3d,
    /// Takes a vector from the cam0 frame to the cam1 frame
    pub right_orientation: Matrix3d,
    pub right_position: Vector3d,
}

#[derive(Debug)]
pub struct CameraPose {
    pub id: usize,
    /// Take a vector from the camera frame to world frame
    pub orientation: Matrix3d,
    pub position: Vector3d,
}

impl CameraState {
    pub fn new(id: usize, left_orientation: Matrix3d, left_position: Vector3d, r_cam0_cam1: &Matrix3d, t_cam0_cam1: &Vector3d) -> Self {
        let right_orientation =  left_orientation * r_cam0_cam1.transpose();
        let right_position = -left_orientation * r_cam0_cam1.transpose() * t_cam0_cam1 + left_position; 
    
        Self {
            id, 
            left_orientation, 
            left_position, 
            right_orientation, 
            right_position, 
        }
    }

    pub fn convert_to_stereo_poses_vec(&self) -> Vec<CameraPose> {
        let left_pose = CameraPose {
            id: self.id, 
            orientation: self.left_orientation, 
            position: self.left_position, 
        }; 
        let right_pose = CameraPose {
            id: self.id, 
            orientation: self.right_orientation, 
            position: self.right_position, 
        }; 
        vec![left_pose, right_pose]
    }
}
