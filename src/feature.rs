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
