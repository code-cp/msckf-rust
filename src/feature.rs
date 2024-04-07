use crate::camera::CameraState;
use crate::my_types::*;

#[derive(Clone, Copy, Debug)]
pub struct Feature {
    pub point: Vector2d,
    pub id: TrackId,
}

#[derive(Clone, Debug)]
pub struct TrackPoint {
    pub coordinates: [Vector2d; 2],
    pub normalized_coordinates: [Vector2d; 2],
    pub frame_number: usize,
}

#[derive(Clone, Debug)]
pub struct Track {
    pub points: Vec<TrackPoint>,
    pub id: TrackId,
    pub last_seen: usize,
}

impl Track {
    pub fn new(
        features: [Feature; 2],
        normalized_coordinates: [Vector2d; 2],
        last_seen: usize,
        frame_number: usize,
    ) -> Self {
        assert_eq!(features[0].id, features[1].id);
        Track {
            points: vec![TrackPoint {
                coordinates: [features[0].point, features[1].point],
                normalized_coordinates,
                frame_number,
            }],
            id: features[0].id,
            last_seen,
        }
    }
}

// Algorithm from the book Computer Vision: Algorithms and Applications
// by Richard Szeliski. Chapter 7.1 Triangulation, page 345.
//
// There are many algorithms for triangulation using N cameras. This one is
// probably the simplest to differentiate wrt to all the pose variables.
// However, it has a particular weakness in that it ignores the fact that the
// fixed transformation between the stereo cameras is known, and instead treats
// all the camera rays as equal. Using this triangulation function may degrade
// quality of the visual updates considerably.
//
// NOTE This function is heavily based on the HybVIO implementation here:
//   <https://github.com/SpectacularAI/HybVIO/blob/main/src/odometry/triangulation.cpp>
//   (see `triangulateLinear()`)
pub fn triangulate(
    normalized_coordinates: &[[Vector2d; 2]],
    camera_poses: &[&CameraState],
) -> Option<Vector3d> {
    assert_eq!(normalized_coordinates.len(), camera_poses.len());

    let mut s = Matrix3d::zeros();
    let mut t = Vector3d::zeros();

    for i in 0..normalized_coordinates.len() {
        let camera_poses = camera_poses[i].convert_to_stereo_poses_vec(); 
        for j in 0..2 {
            let pose = &camera_poses[j]; 
            let ip = &normalized_coordinates[i][j];
            let ip = Vector3d::new(ip[0], ip[1], 1.);
            // NOTE, need to use normalize, since vj is a unit vector for direction 
            let vj = pose.orientation * ip.normalize();
            let a = Matrix3d::identity() - vj * vj.transpose();
            s += a;
            t += a * pose.position;
        }
    }

    let inv_s = s.try_inverse()?;
    if inv_s[(0,0)].is_nan() {
        return None; 
    }
    let position = inv_s * t;

    Some(position)
}
