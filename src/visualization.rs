use cv2::prelude::*;
use opencv as cv2;

use anyhow::Result;
use nalgebra as na;
use ndarray as nd;

use crate::dataset::InputFrame;
use crate::feature::{Feature, Track};
use crate::image::Image;

trait AsArray {
    fn try_as_array(&self) -> Result<nd::Array3<u8>>;
}

impl AsArray for cv2::core::Mat {
    fn try_as_array(&self) -> Result<nd::Array3<u8>> {
        let bytes = self.data_bytes()?;
        let size = self.size()?;
        let a = nd::ArrayView3::from_shape((size.height as usize, size.width as usize, 3), bytes)?;
        Ok(a.to_owned())
    }
}

pub fn grayscale_to_cv_8u(img: &Image) -> cv2::core::Mat {
    unsafe {
        cv2::core::Mat::new_rows_cols_with_data(
            img.height as i32,
            img.width as i32,
            cv2::core::CV_8U,
            std::mem::transmute(img.data.as_ptr()),
            cv2::core::Mat_AUTO_STEP,
        )
        .unwrap()
    }
}

pub fn visualize_detected_features(
    frame: &InputFrame,
    features: &Vec<Feature>,
) -> Result<nd::Array3<u8>> {
    let cv_mat = grayscale_to_cv_8u(frame.images[0]);
    let mut color_mat: cv2::core::Mat = cv_mat.to_owned();
    cv2::imgproc::cvt_color(&cv_mat, &mut color_mat, cv2::imgproc::COLOR_GRAY2RGB, 0)?;

    for feature in features.iter() {
        let point_cv = cv2::core::Point {
            x: feature.point.x as i32,
            y: feature.point.y as i32,
        };
        cv2::imgproc::draw_marker(
            &mut color_mat,
            point_cv,
            cv2::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            1,
            1,
            1,
        )?;
    }

    color_mat.try_as_array()
}

pub fn visualize_tracked_features(
    frame: &InputFrame,
    tracks: &Vec<Track>,
) -> Result<nd::Array3<u8>> {
    let cv_mat = grayscale_to_cv_8u(frame.images[0]);
    let mut color_mat: cv2::core::Mat = cv_mat.to_owned();
    cv2::imgproc::cvt_color(&cv_mat, &mut color_mat, cv2::imgproc::COLOR_GRAY2RGB, 0)?;

    for track in tracks.iter() {
        let mut previous_point = None; 
        for (idx, point) in track.points.iter().enumerate() {
            let current_point_cv = cv2::core::Point {
                x: point.coordinates[0][0] as i32,
                y: point.coordinates[0][1] as i32,
            };
            let marker_size = if idx == 0 {3} else {1}; 
            cv2::imgproc::draw_marker(
                &mut color_mat,
                current_point_cv,
                cv2::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                1,
                marker_size,
                1,
                1,
            )?;

            if let Some(previous_point_cv) = previous_point {
                cv2::imgproc::line(&mut color_mat, previous_point_cv, current_point_cv, cv2::core::Scalar::new(0.0,0.0,255.0, 0.0), 1, 1, 0)?;
            }

            previous_point = Some(current_point_cv); 
        }
    }

    color_mat.try_as_array()
}
