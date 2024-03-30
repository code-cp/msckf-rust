use anyhow::Result;
use nalgebra as na;

use crate::camera::Camera;
use crate::feature::Feature;
use crate::frame::PyramidFrame;
use crate::image::*;
use crate::my_types::*;
use crate::pyramid::Pyramid;

type Range = [[i16; 2]; 2];

/// average depth of the scene
const AVERAGE_DISTANCE_METERS: f64 = 5.;

pub struct OpticalFlow {
    lk_iters: usize,
    lk_levels: usize,
    lk_win_size: usize,
    lk_term: f64,
    lk_min_eig: f64,
    lk_epipolar_max_dist: f64,
    ix: Matrixd,
    iy: Matrixd,
    it: Matrixd,
    grid: Matrixd,
    cam0_to_cam1: Matrix4d,
}

#[derive(Clone, Copy, PartialEq)]
pub enum OpticalFlowKind {
    LeftPreviousToCurrent,
    LeftCurrentToRightCurrent,
    LeftCurrentToRightCurrentDetection,
}

impl OpticalFlow {
    pub fn new(cam0_to_cam1: Matrix4d) -> Result<OpticalFlow> {
        let lk_levels = 3;
        let lk_iters = 5;
        let lk_win_size = 7;
        let lk_term = 0.1;
        let lk_min_eig = 1e-4;
        let lk_epipolar_max_dist = 2.0;

        Ok(OpticalFlow {
            lk_iters,
            lk_levels,
            lk_win_size,
            lk_term,
            lk_min_eig,
            lk_epipolar_max_dist,
            ix: na::DMatrix::zeros(lk_win_size, lk_win_size),
            iy: na::DMatrix::zeros(lk_win_size, lk_win_size),
            it: na::DMatrix::zeros(lk_win_size, lk_win_size),
            grid: na::DMatrix::zeros(lk_win_size, lk_win_size),
            cam0_to_cam1,
        })
    }

    pub fn process(
        &mut self,
        kind: OpticalFlowKind,
        pyramid_frame0: &PyramidFrame,
        pyramid_frame1: &PyramidFrame,
        cameras: &[&Camera],
        features0_in: &[Feature],
        features0: &mut Vec<Feature>,
        features1: &mut Vec<Feature>,
    ) {
        let lk_epipolar_max_dist2 =
            (pyramid_frame0.image.scale() * self.lk_epipolar_max_dist).powi(2);

        // Successfully tracked features
        features0.clear();
        features1.clear();

        for feature0 in features0_in {
            let point1_in = compute_initial_guess(feature0.point, cameras, &self.cam0_to_cam1);
            let feature1 =
                self.process_feature(pyramid_frame0, pyramid_frame1, *feature0, point1_in);
            let feature1 = if let Some(feature1) = feature1 {
                feature1
            } else {
                continue;
            };
            if !epipolar_check(
                &feature0,
                &feature1,
                kind,
                cameras,
                &self.cam0_to_cam1,
                lk_epipolar_max_dist2,
            ) {
                continue;
            }
            features1.push(feature1);
            features0.push(*feature0);
        }
    }

    /// ref http://robots.stanford.edu/cs223b04/algo_tracking.pdf
    fn process_feature(
        &mut self,
        pyramid_frame0: &PyramidFrame,
        pyramid_frame1: &PyramidFrame,
        feature0: Feature,
        point1_in: Option<Vector2d>,
    ) -> Option<Feature> {
        let lk_term2 = self.lk_term.powi(2);
        let r = (self.lk_win_size - 1) / 2;

        // initial guess
        let mut g_init = point1_in
            .map(|p| p - feature0.point)
            .unwrap_or(Vector2d::zeros())
            / u32::pow(2, self.lk_levels as u32) as f64;
        let mut d = Vector2d::zeros();
        for level in (0..self.lk_levels + 1).rev() {
            let level0 = pyramid_frame0.get_image_at_level(level);
            let level1 = pyramid_frame1.get_image_at_level(level);
            let u = feature0.point / u32::pow(2, level as u32) as f64;
            let range = integration_range(&level0, u, r, 1)?;
            // compute the derivative
            scharr(
                &level0,
                u,
                range,
                &mut self.ix,
                &mut self.iy,
                &mut self.grid,
            );
            let gradient = spatial_gradient(&self.ix, &self.iy);
            if gradient.eigenvalues()?.min() < self.lk_min_eig {
                return None;
            }
            let mut converged = false;
            let mut nu = Vector2d::zeros();
            for _ in 0..self.lk_iters {
                image_difference(range, r, &self.grid, &mut self.it, &level1, u + g_init + nu)?;
                let eta = flow_vector(&gradient, &self.ix, &self.iy, &self.it)?;
                nu += eta;
                if eta.norm_squared() < lk_term2 {
                    converged = true;
                    break;
                }
            }

            d = nu;
            if !converged {
                return None;
            }
            if level > 0 {
                g_init = 2. * (g_init + d)
            }
        }

        Some(Feature {
            point: feature0.point + g_init + d,
            id: feature0.id,
        })
    }
}

fn epipolar_check(
    feature0: &Feature,
    feature1: &Feature,
    kind: OpticalFlowKind,
    cameras: &[&Camera],
    cam0_to_cam1: &Matrix4d,
    max_dist2: f64,
) -> bool {
    // only check when using stereo images
    if kind != OpticalFlowKind::LeftCurrentToRightCurrent
        && kind != OpticalFlowKind::LeftCurrentToRightCurrentDetection
    {
        return true;
    }

    // Compute a curve in image1 where feature1 should be found
    let curve_point_count = 8;
    let mut curve = vec![];
    curve.clear();
    if let Some(ray) = cameras[0].model.back_project(feature0.point) {
        // s is the depth of point
        let mut s = 0.5;
        for _ in 0..curve_point_count {
            let r0 = s * ray;
            let r1 = transform_vector3d(&cam0_to_cam1, &r0);
            if let Some(pixel) = cameras[1].model.project(r1) {
                curve.push(pixel);
                s *= 2.;
            }
        }
    }

    // Check if feature1 is on the curve.
    // Reverse iterate because the feature is more likely near the "far end"
    for c in curve.iter().rev() {
        if (c - feature1.point).norm_squared() < max_dist2 {
            return true;
        }
    }

    // check if feature1 is close to the curve by interpolation
    for i in 1..curve.len() {
        let c0 = &curve[i - 1];
        let c1 = &curve[i];
        let s2 = (c1 - c0).norm_squared();
        let t = (feature1.point - c0).dot(&(c1 - c0)) / s2;
        if t > 0. && t < 1. && (feature1.point - (c0 + t * (c1 - c0))).norm_squared() < max_dist2 {
            return true;
        }
    }

    false
}

fn flow_vector(gradient: &Matrix2d, ix: &Matrixd, iy: &Matrixd, it: &Matrixd) -> Option<Vector2d> {
    let mut b = Vector2d::zeros();

    for y in 0..iy.nrows() {
        for x in 0..ix.ncols() {
            b[0] += it[(y, x)] * ix[(y, x)];
            b[1] += it[(y, x)] * iy[(y, x)];
        }
    }

    gradient.try_inverse().map(|inv_g| inv_g * b)
}

fn image_difference(
    prev_range: Range,
    r: usize,
    i0: &Matrixd,
    mut it: &mut Matrixd,
    level: &Image,
    center: Vector2d,
) -> Option<()> {
    let range = integration_range(level, center, r, 0)?;
    if range != prev_range {
        return None;
    }
    fill_grid(level, range, center, &mut it);
    *it *= -1.;
    *it += i0.slice((1, 1), (it.nrows(), it.ncols()));
    Some(())
}

fn spatial_gradient(ix: &Matrixd, iy: &Matrixd) -> Matrix2d {
    assert_eq!(ix.nrows(), iy.nrows());
    assert_eq!(ix.ncols(), iy.ncols());

    let mut x2 = 0.;
    let mut y2 = 0.;
    let mut xy = 0.;

    for y in 0..iy.nrows() {
        for x in 0..ix.ncols() {
            x2 += ix[(y, x)] * ix[(y, x)];
            y2 += iy[(y, x)] * iy[(y, x)];
            xy += ix[(y, x)] * iy[(y, x)];
        }
    }

    Matrix2d::new(x2, xy, xy, y2)
}

/// ref https://theailearner.com/tag/scharr-operator/
fn scharr(
    level: &Image,
    center: Vector2d,
    range: Range,
    out_x: &mut Matrixd,
    out_y: &mut Matrixd,
    mut grid: &mut Matrixd,
) {
    let grange = [
        [range[0][0] - 1, range[0][1] + 1],
        [range[1][0] - 1, range[1][1] + 1],
    ];
    fill_grid(level, grange, center, &mut grid);

    *out_x = Matrixd::zeros(grid.nrows() - 2, grid.ncols() - 2);
    *out_y = Matrixd::zeros(grid.nrows() - 2, grid.ncols() - 2);
    for y in 1..(grid.nrows() - 1) {
        for x in 1..(grid.ncols() - 1) {
            out_x[(y - 1, x - 1)] =
                (10. * grid[(y, x + 1)] + 3. * grid[(y + 1, x + 1)] + 3. * grid[(y - 1, x + 1)]
                    - 10. * grid[(y, x - 1)]
                    - 3. * grid[(y + 1, x - 1)]
                    - 3. * grid[(y - 1, x - 1)])
                    / 32.;
            out_y[(y - 1, x - 1)] =
                (10. * grid[(y + 1, x)] + 3. * grid[(y + 1, x + 1)] + 3. * grid[(y + 1, x - 1)]
                    - 10. * grid[(y - 1, x)]
                    - 3. * grid[(y - 1, x + 1)]
                    - 3. * grid[(y - 1, x - 1)])
                    / 32.;
        }
    }
}

fn fill_grid(level: &Image, range: Range, center: Vector2d, grid: &mut Matrixd) {
    *grid = na::DMatrix::zeros(
        (range[1][1] - range[1][0] + 1) as usize,
        (range[0][1] - range[0][0] + 1) as usize,
    );

    for (y_ind, y) in (range[1][0]..=range[1][1]).enumerate() {
        for (x_ind, x) in (range[0][0]..=range[0][1]).enumerate() {
            grid[(y_ind, x_ind)] = bilinear(level, center + Vector2d::new(x as f64, y as f64));
        }
    }
}

/// Returns closed range of integer steps that can be taken without going outside
/// the image borders. Returns None if the center point is outside the level
/// boundaries.
fn integration_range(level: &Image, center: Vector2d, r: usize, padding: i16) -> Option<Range> {
    let r = r as i16;
    let mut range = [[0, 0], [0, 0]];
    for i in 0..2 {
        let s = if i == 0 { level.width } else { level.height };
        if center[i] < 0. || center[i] > (s - 1) as f64 {
            return None;
        }
        let n = center[i] as i16;
        let fract = if center[i].fract() > 0. { 1 } else { 0 };
        range[i] = [
            i16::max(-r, -n + padding),
            i16::min(r, s as i16 - n - padding - 1 - fract),
        ]
    }
    Some(range)
}

fn compute_initial_guess(
    p0: Vector2d,
    cameras: &[&Camera],
    cam0_to_cam1: &Matrix4d,
) -> Option<Vector2d> {
    if let Some(ray) = cameras[0].model.back_project(p0) {
        let r0 = AVERAGE_DISTANCE_METERS * ray;
        let r1 = transform_vector3d(&cam0_to_cam1, &r0);
        cameras[1].model.project(r1)
    } else {
        None
    }
}

fn transform_vector3d(m: &Matrix4d, v: &Vector3d) -> Vector3d {
    m.fixed_slice::<3, 3>(0, 0) * v + m.fixed_slice::<3, 1>(0, 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pyramid_frame(image: Image, lk_levels: usize) -> PyramidFrame {
        let mut pyramid = Pyramid::empty();
        Pyramid::compute(&mut pyramid, &image, lk_levels).unwrap();
        PyramidFrame { image, pyramid }
    }

    #[test]
    fn test_flow() {
        let image_size: i32 = 128;
        let mut image0 = Image {
            data: vec![0; image_size.pow(2) as usize],
            width: 128,
            height: 128,
        };
        let mut image1 = image0.clone();

        let patch = Image {
            data: vec![
                44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 55, 55, 55, 55, 55, 55, 55, 44, 44, 55, 77,
                77, 77, 77, 77, 55, 44, 44, 55, 77, 88, 88, 88, 77, 55, 44, 44, 55, 77, 88, 99, 88,
                77, 55, 44, 44, 55, 77, 88, 88, 88, 77, 55, 44, 44, 55, 77, 77, 77, 77, 77, 55, 44,
                44, 55, 55, 55, 55, 55, 55, 55, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
            ],
            width: 9,
            height: 9,
        };

        // Place the patch at different positions in the two images.
        let x: i32 = 60;
        let y: i32 = 60;
        let dx: i32 = -14;
        let dy: i32 = 7;
        image0.set_sub_image_i32(x, y, &patch);
        image1.set_sub_image_i32(x + dx, y + dy, &patch);

        let lk_levels = 3;

        let pyramid_frame0 = make_pyramid_frame(image0, lk_levels);
        let pyramid_frame1 = make_pyramid_frame(image1, lk_levels);

        // Place feature at center of the first patch.
        let r = (patch.width - 1) as i32 / 2;
        let feature0 = Feature {
            point: Vector2d::new((x + r) as f64, (y + r) as f64),
            id: TrackId(0),
        };

        let dummy_cam0_to_cam1 = Matrix4d::zeros();
        let mut flow = OpticalFlow::new(dummy_cam0_to_cam1).unwrap();
        if let Some(feature1) =
            flow.process_feature(&pyramid_frame0, &pyramid_frame1, feature0, None)
        {
            let err = (feature1.point - feature0.point) - Vector2d::new(dx as f64, dy as f64);
            println!("optical flow err {}", err.norm());
        } else {
            assert!(false);
        }
    }

    #[test]
    fn test_scharr() {
        let mut image = Image {
            data: vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            width: 5,
            height: 5,
        };

        let mut out_x = na::dmatrix!();
        let mut out_y = na::dmatrix!();
        let mut grid = na::dmatrix!();

        let center = Vector2d::new(2.0, 2.0);
        let range = integration_range(&image, center, 1, 1).unwrap();
        scharr(&image, center, range, &mut out_x, &mut out_y, &mut grid);
        assert_eq!(out_x, na::DMatrix::zeros(3, 3));
        assert_eq!(out_y, na::DMatrix::zeros(3, 3));

        image.data = vec![
            0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
        ];
        scharr(&image, center, range, &mut out_x, &mut out_y, &mut grid);
        assert_eq!(out_x, na::DMatrix::repeat(3, 3, 1.));
        assert_eq!(out_y, na::DMatrix::zeros(3, 3));

        image.data = vec![
            0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8,
        ];
        scharr(&image, center, range, &mut out_x, &mut out_y, &mut grid);
        assert_eq!(out_x, na::DMatrix::repeat(3, 3, 1.));
        assert_eq!(out_y, na::DMatrix::repeat(3, 3, 1.));

        image.data = vec![
            0, 0, 5, 0, 0,
            0, 0, 5, 0, 0,
            0, 0, 5, 0, 0,
            0, 0, 5, 0, 0,
            0, 0, 5, 0, 0,
          ];
          scharr(&image, center, range, &mut out_x, &mut out_y, &mut grid);
          let answer_x = na::dmatrix!(
            2.5, 0., -2.5;
            2.5, 0., -2.5;
            2.5, 0., -2.5;
          );
          assert_eq!(out_x, answer_x);
          assert_eq!(out_y, na::DMatrix::zeros(3, 3));
    }

    #[test]
    fn test_integration_range() {
      // Width and height are pixels. Coordinate (0, 0) means center of top-left
      // pixel. Thus (9, 9) is the center of the bottom-right pixel for 10x10
      // image.
      let image = Image {
        data: vec![],
        width: 10,
        height: 10,
      };
      assert_eq!(integration_range(&image, Vector2d::new(4.5, 4.5), 3, 0).unwrap(), [[-3, 3], [-3, 3]]);
      assert_eq!(integration_range(&image, Vector2d::new(1.5, 2.5), 3, 0).unwrap(), [[-1, 3], [-2, 3]]);
      assert_eq!(integration_range(&image, Vector2d::new(1.0, 2.0), 3, 0).unwrap(), [[-1, 3], [-2, 3]]);
      assert_eq!(integration_range(&image, Vector2d::new(0.9, 1.9), 3, 0).unwrap(), [[0, 3], [-1, 3]]);
      assert_eq!(integration_range(&image, Vector2d::new(0.9, 1.9), 3, 1).unwrap(), [[1, 3], [0, 3]]);
      assert_eq!(integration_range(&image, Vector2d::new(8.5, 2.0), 3, 0).unwrap(), [[-3, 0], [-2, 3]]);
      assert_eq!(integration_range(&image, Vector2d::new(9.5, 2.0), 3, 0), None);
    }
}
