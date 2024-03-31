use nalgebra as na;

use crate::image::Image;

pub type Vector2d = na::Vector2<f64>;
pub type Vector3d = na::Vector3<f64>;
pub type Vector4d = na::Vector4<f64>;
pub type Vectord = nalgebra::DVector<f64>;

pub type Matrixd = nalgebra::DMatrix<f64>;
pub type Matrix2d = nalgebra::Matrix2<f64>; 
pub type Matrix3d = nalgebra::Matrix3<f64>;
pub type Matrix4d = nalgebra::Matrix4<f64>;
pub type Matrix34d = nalgebra::Matrix3x4<f64>;
pub type Matrix23d = nalgebra::Matrix2x3<f64>;

pub type ImageShape = (i32, i32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackId(pub usize);
