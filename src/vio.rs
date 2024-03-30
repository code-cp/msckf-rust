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

use crate::dataset::*;
use crate::image::*;
use crate::kalman_filter::*;
use crate::my_types::*;
use crate::camera::*; 

#[derive(Debug)]
pub struct VIO {
    /// The time of the last sensor data
    last_time: Option<f64>,
    last_gyro: Option<(f64, Vector3d)>,
    last_acc: Option<(f64, Vector3d)>,
    recorder: RecordingStream,
    orientation_initialized: bool,
    imu_state: StateServer,
    frames: VecDeque<StereoImage>,
    cameras: Vec<Camera>,
}

impl VIO {
    pub fn new(cameras: Vec<Camera>, extrinsics: &Extrinsics) -> Self {
        // visualization
        let recorder = RecordingStreamBuilder::new("msckf")
            .save("./logs/my_recording.rrd")
            .unwrap();
        let imu_state = StateServer::new(extrinsics);

        Self {
            last_time: None,
            last_acc: None,
            last_gyro: None,
            recorder,
            orientation_initialized: false,
            imu_state,
            frames: VecDeque::new(),
            cameras, 
        }
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
        self.imu_state.predict(time, gyro, acc);
    }

    pub fn process_frame(&mut self, frame: &InputFrame) -> Result<()> {
        // TODO: implement frame processing
        if self.frames.len() == 2 {
            self.frames.pop_front();
        }
        self.frames
            .push_back((frame.images[0].clone(), frame.images[1].clone()));

        let image0_nd = nd::Array::from_shape_vec(
            (frame.images[0].height, frame.images[0].width),
            frame.images[0].clone().data,
        )
        .unwrap();

        let image0_rerun = rerun::Image::try_from(image0_nd.clone())?;
        self.recorder.log("world/camera/image", &image0_rerun)?;

        Ok(())
    }
}
