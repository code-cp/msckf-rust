use nalgebra::matrix; 

use crate::my_types::*;
use crate::camera::*; 

#[derive(Debug)]
pub struct PinholeModel {
    pub camera_matrix: Matrix3d,
    pub camera_matrix_inv: Matrix3d,
}

impl PinholeModel {
    pub fn new(camera_matrix: Matrix3d) -> Self {
        PinholeModel {
            camera_matrix,
            camera_matrix_inv: camera_matrix.try_inverse().unwrap(),
        }
    }
}

impl CameraModel for PinholeModel {
    fn back_project(&self, pixel: Vector2d) -> Option<Vector3d> {
        let p = Vector2d::new(
            (pixel[0] - self.camera_matrix[(0, 2)]) / self.camera_matrix[(0, 0)],
            (pixel[1] - self.camera_matrix[(1, 2)]) / self.camera_matrix[(1, 1)],
        );
        Some(Vector3d::new(p[0], p[1], 1.))
    }

    fn project_with_derivative(
        &self,
        ray: Vector3d,
        compute_derivative: bool,
    ) -> (Option<Vector2d>, Option<Matrix23d>) {
        // ray is behind camera
        if ray[2] <= 0. {
            return (None, None);
        }
        let z_inv = 1. / ray[2];
        let p = Vector3d::new(z_inv * ray[0], z_inv * ray[1], 1.);
        let pixel = self.camera_matrix * p;
        // derivative of uv wrt xyz
        let pixel_derivative = if compute_derivative {
            // derivative of normalized uv wrt xyz
            let x_xbar_derivative = matrix!(
                z_inv,
                0.,
                -ray[0] * z_inv.powi(2);
                0.,
                z_inv,
                -ray[1] * z_inv.powi(2);
            );
            Some(self.camera_matrix.fixed_view::<2, 2>(0, 0) * x_xbar_derivative)
        } else {
            None
        };
        (Some(Vector2d::new(pixel[0], pixel[1])), pixel_derivative)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinhole() {
        let intrinsics = Matrix3d::new(1000., 0., 360., 0., 1000., 240., 0., 0., 1.);
        let ray = Vector3d::new(-0.25, 0.11, 2.).normalize();
        let camera = PinholeModel::new(intrinsics); 
        let pixel = camera.project(ray).unwrap();
        assert!((pixel - Vector2d::new(235., 695.)).norm() < 1e-6);

        let intrinsics = Matrix3d::new(458., 0., 367.215, 0., 458., 248.375, 0., 0., 1.);
        let camera = PinholeModel::new(intrinsics);
        let pixel = camera.project(ray).unwrap();
        assert!((pixel - Vector2d::new(310.26612557476517, 273.4325047471033)).norm() < 1e-6);

        let ray_computed = camera.back_project(pixel).unwrap();
        assert!((ray_computed - ray).norm() < 1e-10)
    }
}
