use crate::my_types::*; 
use crate::detector::*; 
use crate::optical_flow::OpticalFlow; 
use crate::feature::*; 

pub struct Tracker {
    detector: Detector,
    optical_flow: OpticalFlow,
    tracks: Vec<Track>,
    max_tracks: usize,
    next_id: TrackId,
    step: usize,
}
