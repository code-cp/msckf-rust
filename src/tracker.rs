pub struct Tracker {
    detector: Detector,
    optical_flow: OpticalFlow,
    tracks: Vector<Track>,
    max_tracks: usize,
    next_id: TrackId,
    step: usize,
}
