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
    // Take a vector from the camera frame to world frame
    pub orientation: Matrix3d,
    pub position: Vector3d,
}
