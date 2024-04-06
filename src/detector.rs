use crate::feature::Feature;
use crate::image::Image;
use crate::my_types::*;

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

#[derive(Debug)]
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

    pub fn process(
        &mut self,
        image: &Image,
        detections: &mut Vec<Feature>,
        required_feature_count: usize,
        next_id: &mut TrackId,
    ) {
        assert!(image.width > 1 + 2 * CIRCLE_RADIUS);
        assert!(image.height > 1 + 2 * CIRCLE_RADIUS);

        debug!("image width {} height {}", image.width, image.height); 

        detections.clear();

        if required_feature_count == 0 {
            return;
        }

        self.mask.clear();
        self.mask = vec![false; image.width * image.height];

        let mut threshold = self.start_threshold;
        let mask_radius = ((image.width.max(image.height) as f32) / 100.0).round() as i32;
        let threshold_halving_iterations = 4;

        'detection: for _ in 0..threshold_halving_iterations {
            for x in CIRCLE_RADIUS..(image.width - CIRCLE_RADIUS) {
                for y in CIRCLE_RADIUS..(image.height - CIRCLE_RADIUS) {
                    if self.mask[y * image.width + x] {
                        continue;
                    }
                    if !self.detect_at_pixel(x as i32, y as i32, image, threshold) {continue;}
                    detections.push(Feature {
                        point: Vector2d::new(x as f64, y as f64),
                        id: *next_id,
                    });
                    next_id.0 += 1;

                    add_mask(
                        &mut self.mask,
                        x as i32,
                        y as i32,
                        image.width,
                        image.height,
                        mask_radius,
                    );
                    if detections.len() >= required_feature_count {
                        break 'detection;
                    }
                }
                threshold /= 2;
            }
        }
    }

    pub fn detect_at_pixel(&mut self, x: i32, y: i32, image: &Image, threashold: i16) -> bool {
        let center_value = image.value_i32(x, y) as i16;
        if continuous(x, y, image, |v| (v as i16) < center_value - threashold) {
            return true;
        }
        if continuous(x, y, image, |v| (v as i16) > center_value + threashold) {
            return true;
        }
        false
    }
}

fn continuous<F: Fn(u8) -> bool>(x: i32, y: i32, image: &Image, f: F) -> bool {
    // quickly reject the corner point
    if FAST_VARIANT_N >= 9 && !f(image.value_i32(x + 3, y)) && !f(image.value_i32(x - 3, y)) {
        return false;
    }

    let it = CircleIterator::new(x, y);
    let mut n = 0;
    for p in it {
        // check continous FAST_VARIANT_N points
        if f(image.value_i32(p[0], p[1])) {
            n += 1;
            if n >= FAST_VARIANT_N {
                return true;
            }
        } else {
            n = 0;
        }
    }

    false
}

struct CircleIterator {
    center: [i32; 2],
    ind: usize,
}

impl CircleIterator {
    pub fn new(x: i32, y: i32) -> Self {
        Self {
            center: [x, y],
            ind: 0,
        }
    }
}

impl Iterator for CircleIterator {
    type Item = [i32; 2];

    fn next(&mut self) -> Option<Self::Item> {
        if self.ind >= CIRCLE.len() {
            return None;
        }
        self.ind += 1;
        Some([
            self.center[0] + CIRCLE[self.ind - 1][0],
            self.center[1] + CIRCLE[self.ind - 1][1],
        ])
    }
}

/// This mask prevents detecting features too close to each other, ie nms 
fn add_mask(mask: &mut Vec<bool>, cx: i32, cy: i32, width: usize, height: usize, mask_radius: i32) {
    for x in (cx - mask_radius)..(cx + mask_radius + 1) {
        if x < 0 || x >= width as i32 {
            continue;
        }
        for y in (cy - mask_radius)..(cy + mask_radius + 1) {
            if y < 0 || y >= height as i32 {
                continue;
            }
            mask[y as usize * width + x as usize] = true;
        }
    }
}
