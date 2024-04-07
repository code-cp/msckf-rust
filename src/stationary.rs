
use crate::feature::Track; 

#[derive(Debug)]
pub struct Stationary {
    max_displacement: f64
}

impl Stationary {
    pub fn new() -> Self {
        let max_displacement = 1_f64.powi(2); 
        Self {
            max_displacement 
        }
    }

    pub fn is_static(&self, tracks: &[Track]) -> bool {
        for track in tracks {
            if track.points.len() < 2 {continue}
            let mut it = track.points.iter().rev(); 
            // Compare two last points from the first stereo camera
            let p0 = it.next().unwrap().coordinates[0];
            let p1 = it.next().unwrap().coordinates[0]; 
            if (p0 - p1).norm_squared() > self.max_displacement {return false}
        }
        true 
    }
}