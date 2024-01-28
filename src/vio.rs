use anyhow::{anyhow, bail, Result, Context as AnyhowContext}; 
use log::{debug, error, info, warn, LevelFilter};
use ndarray as nd;
use rerun::{RecordingStream, RecordingStreamBuilder}; 

use crate::dataset::*; 
use crate::kalman_filter;
use crate::my_types::*; 
use crate::kalman_filter::*; 

pub struct VIO {
    /// The time of the last sensor data 
    last_time: Option<f64>, 
    last_gyro: Option<(f64, Vector3d)>, 
    last_acc: Option<(f64, Vector3d)>,
    recorder: RecordingStream,
    orientation_initialized: bool,  
    kalman_filter: KalmanFilter,
}

impl VIO {
    pub fn new() -> Self {
        // visualization 
        let recorder = RecordingStreamBuilder::new("msckf").save("./logs/my_recording.rrd").unwrap();
        let kalman_filter = KalmanFilter::new(); 

        Self {
            last_time: None, 
            last_acc: None, 
            last_gyro: None, 
            recorder,  
            orientation_initialized: false, 
            kalman_filter, 
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

                let image0_nd = nd::Array::from_shape_vec(
                    (frame.images[0].height, frame.images[0].width), 
                    frame.images[0].clone().data,  
                ).unwrap();

                let image0_rerun = rerun::Image::try_from(image0_nd.clone())?;
                self.recorder.log("world/camera/image", &image0_rerun)?;
            }, 
            InputSensor::Gyroscope(gyro) => {
                self.last_gyro = Some((data.time, gyro)); 
            }, 
            InputSensor::Accelerometer(acc) => {
                self.last_acc = Some((data.time, acc)); 
            }
        }

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
        self.kalman_filter.predict(time, gyro, acc); 
    }
}