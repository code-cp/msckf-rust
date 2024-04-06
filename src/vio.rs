use anyhow::{anyhow, bail, Context as AnyhowContext, Result};
use log::{debug, error, info, warn, LevelFilter};
use ndarray as nd;
use rerun::{RecordingStream, RecordingStreamBuilder};
use std::collections::VecDeque;

use tracing::instrument;
use tracing::Span;
use tracing::{span, Level};
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::camera::*;
use crate::dataset::*;
use crate::frame::Frame;
use crate::kalman_filter::*;
use crate::my_types::*;
use crate::tracker::Tracker;
use crate::visualization::*;

#[derive(Debug)]
pub struct VIO {
    /// The time of the last sensor data
    last_time: Option<f64>,
    last_gyro: Option<(f64, Vector3d)>,
    last_acc: Option<(f64, Vector3d)>,
    recorder: RecordingStream,
    orientation_initialized: bool,
    state_server: StateServer,
    frames: VecDeque<Frame>,
    cameras: Vec<Camera>,
    // Incremented just before processing a new frame. 0 before the first frame.
    frame_number: usize,
    tracker: Tracker,
}

impl VIO {
    pub fn new(cameras: Vec<Camera>, extrinsics: &Extrinsics) -> Result<Self> {
        // visualization
        let recorder = RecordingStreamBuilder::new("msckf")
            .save("./logs/my_recording.rrd")
            .unwrap();
        let state_server = StateServer::new(extrinsics);
        let cam0_to_cam1 = extrinsics.trans_cam0_cam1;

        Ok(Self {
            last_time: None,
            last_acc: None,
            last_gyro: None,
            recorder,
            orientation_initialized: false,
            state_server,
            frames: VecDeque::new(),
            cameras,
            frame_number: 0,
            tracker: Tracker::new(cam0_to_cam1)?,
        })
    }

    pub fn process_data(&mut self, data: &SensorData) -> Result<bool> {
        if let Some(last_time) = self.last_time {
            if data.time < last_time {
                warn!("discard unordered sample");
                return Ok(false);
            }
        }
        self.last_time = Some(data.time);

        // println!("Got sensor data at time: {:?}", data.time);
        match data.sensor {
            InputSensor::Frame(ref frame) => {
                if !self.orientation_initialized {
                    return Ok(false);
                }

                // check the image size of stereo images
                assert!(frame.images[0].width > 0 && frame.images[0].height > 0);
                assert!(frame.images[1].width > 0 && frame.images[1].height > 0);

                Span::current().pb_inc(1);

                self.process_frame(frame)?;
            }
            InputSensor::Gyroscope(gyro) => {
                self.last_gyro = Some((data.time, gyro));
            }
            InputSensor::Accelerometer(acc) => {
                self.last_acc = Some((data.time, acc));
            }
        }

        // Very basic sample synchronization that only aims to cover the case that
        // the gyroscope and accelerometer samples are already paired one-to-one in
        // the input data, but it's not known if the accelerometer or gyroscope
        // sample comes first in the stream.
        if let (Some((time_gyro, gyro)), Some((time_acc, acc))) = (self.last_gyro, self.last_acc) {
            if time_acc >= time_gyro {
                self.process_imu(time_gyro, gyro, acc);
                self.orientation_initialized = true;
                // allow reuse of acc data
                self.last_gyro = None;
            }
        }

        return Ok(true);
    }

    pub fn process_imu(&mut self, time: f64, gyro: Vector3d, acc: Vector3d) {
        self.state_server.predict(time, gyro, acc);
    }

    pub fn process_frame(&mut self, frame: &InputFrame) -> Result<()> {
        self.state_server.augment_state();

        let mut unused_frame = None;
        if self.frames.len() == 2 {
            unused_frame = self.frames.pop_front();
        }
        self.frames.push_back(Frame::new(frame, unused_frame)?);

        let frame0 = if self.frames.len() < 2 {
            None
        } else {
            self.frames.front().clone()
        };
        let frame1 = self.frames.back().clone().unwrap();
        self.tracker
            .process(frame0, frame1, &self.cameras, self.frame_number);

        self.state_server.prune_state();

        // show detected features
        let feature_detection_image = visualize_detected_features(frame, &self.tracker.features0)?;
        self.recorder.log(
            "world/camera/feature_detection",
            &rerun::Image::try_from(feature_detection_image)?,
        )?;

        // show feature tracks
        // let feature_track_image = visualize_tracked_features(frame, &self.tracker.tracks)?;
        // self.recorder.log(
        //     "world/camera/feature_tracks",
        //     &rerun::Image::try_from(feature_track_image)?,
        // )?;

        // show feature map
        // self.recorder.log(
        //     "world/feature_map",
        //     &rerun::Points3D::new(self.state_server.get_feature_map_for_visualization().iter()),
        // )?;

        // show camera pose
        // let w_tr_c = self.state_server.get_camera_pose();
        // let arrow = rerun::Arrows3D::from_vectors(
        //     [(w_tr_c.rotation.euler_angles().0 as f32, w_tr_c.rotation.euler_angles().1 as f32, w_tr_c.rotation.euler_angles().2 as f32)]
        //     ).
        //     with_origins([(
        //         w_tr_c.translation.x as f32,
        //         w_tr_c.translation.y as f32,
        //         w_tr_c.translation.z as f32,
        //     )]);
        // self.recorder.log("world/camera/pose", &arrow)?;

        Ok(())
    }
}
