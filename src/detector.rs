const FAST_VARIANT_N: usize = 12;

/// A Bresenham circle.
/// ref https://en.wikipedia.org/wiki/Midpoint_circle_algorithm
const CIRCLE_RADIUS: usize = 3;
const CIRCLE: [[i32; 2]; 16] = [
    [0, -3],
    [1, -3],
    [2, -2],
    [3, -1],
    [3, 0],
    [3, 1],
    [2, 2],
    [1, 3],
    [0, 3],
    [-1, 3],
    [-2, 2],
    [-3, 1],
    [-3, 0],
    [-3, -1],
    [-2, -2],
    [-1, -3],
];

pub struct Detector {
    start_threshold: i16,
    mask: Vec<bool>,
}

impl Detector {
    pub fn new() -> Self {
        Detector {
            start_threshold: 128,
            mask: vec![],
        }
    }
}
