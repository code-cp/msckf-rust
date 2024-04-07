use anyhow::{anyhow, bail, Context as AnyhowContext, Result};
use log::{debug, error, info, warn, LevelFilter};
use serde_json::{from_str, Value};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::image::Image;
use crate::my_types::*;
use crate::video::VideoInput;
use crate::math::to_rotation_matrix; 

pub struct Dataset {
    reader: BufReader<File>,
    line: String,
    video_inputs: Vec<VideoInput>,
    pub length: u64,
    pub first_pose_gt: Matrix4d, 
}

#[derive(Debug)]
pub struct InputFrame<'a> {
    pub images: Vec<&'a Image>,
}

#[derive(Debug)]
pub enum InputSensor<'a> {
    Gyroscope(Vector3d),
    Accelerometer(Vector3d),
    Frame(InputFrame<'a>),
    /// pose from groundtruth 
    Pose(Matrix4d),
}

#[derive(Debug)]
pub struct SensorData<'a> {
    pub time: f64,
    pub sensor: InputSensor<'a>,
}

fn get_dataset_length(file: File) -> u64 {
    let reader = BufReader::new(file);
    let mut total_items = 0;
    for line in reader.lines() {
        if let Ok(line) = line {
            if let Ok(json) = from_str::<Value>(&line) {
                if let Some(_) = json.get("number") {
                    total_items += 1;
                }
            }
        }
    }
    total_items
}

fn load_pose_from_groundtruth(groundtruth: &Value) -> Matrix4d {
    let mut pose = Matrix4d::zeros(); 

    let position = groundtruth.get("position").unwrap().clone(); 
    let orientation = groundtruth.get("orientation").unwrap().clone(); 

    if let (Some(px), Some(py), Some(pz)) = (position.get("x").and_then(|v| v.as_f64()),
        position.get("y").and_then(|v| v.as_f64()),
        position.get("z").and_then(|v| v.as_f64())) {
        pose.fixed_view_mut::<3,1>(0,3).copy_from(&Vector3d::new(px, py, pz)); 
    }

    if let (Some(x), Some(y), Some(z), Some(w)) = (orientation.get("x").and_then(|v| v.as_f64()),
        orientation.get("y").and_then(|v| v.as_f64()),
        orientation.get("z").and_then(|v| v.as_f64()),
        orientation.get("w").and_then(|v| v.as_f64())) {
        let rotation_matrix = to_rotation_matrix(Vector4d::new(w, x, y, z)); 
        pose.fixed_view_mut::<3,3>(0,0).copy_from(&rotation_matrix); 
    }

    pose
}

pub fn get_first_pose(file: File) -> Matrix4d {
    let reader = BufReader::new(file);
    let mut pose = Matrix4d::zeros(); 
    for line in reader.lines() {
        if let Ok(line) = line {
            if let Ok(json) = from_str::<Value>(&line) {
                if let Some(groundtruth) = json.get("groundTruth") {
                    pose = load_pose_from_groundtruth(groundtruth);
                    break; 
                }
            }
        }
    }

    pose
}

impl Dataset {
    pub fn new(path: &Path) -> Result<Dataset> {
        let file: File = File::open(path.join("data.jsonl"))?;
        let length = get_dataset_length(File::open(path.join("data.jsonl")).unwrap());
        let first_pose_gt = get_first_pose(File::open(path.join("data.jsonl")).unwrap()); 

        let video_inputs = vec![
            VideoInput::new(&path.join("data.mp4"))?,
            VideoInput::new(&path.join("data2.mp4"))?,
        ];

        Ok(Dataset {
            reader: BufReader::new(file),
            line: String::new(),
            video_inputs,
            length,
            first_pose_gt, 
        })
    }

    pub fn next(&mut self) -> Result<Option<SensorData>> {
        loop {
            self.line.clear();
            match self.reader.read_line(&mut self.line) {
                Ok(0) => return Ok(None),
                Err(err) => bail!("Failed to read line {}", err),
                _ => {}
            }

            let value: serde_json::Value = serde_json::from_str(&self.line).context(format!(
                "JSON deserialization failed for line: {}",
                self.line
            ))?;
            let value = value.as_object().ok_or(anyhow!("JSON line is not a map"))?;

            let time = value["time"]
                .as_f64()
                .ok_or(anyhow!("Time is not a number"))?;

            if let Some(sensor) = value.get("sensor") {
                let v = &sensor["values"]
                    .as_array()
                    .ok_or(anyhow!("Sensor values not an array"))?;
                let v: Vec<f64> = v.iter().map(|x| x.as_f64().unwrap()).collect();
                assert!(
                    v.len() >= 3,
                    "Sensor values array must contain at least 3 elements"
                );
                let v = Vector3d::new(v[0], v[1], v[2]);

                let sensor_type = sensor["type"]
                    .as_str()
                    .ok_or(anyhow!("Sensor type missing"))?;
                match sensor_type {
                    "gyroscope" => {
                        return Ok(Some(SensorData {
                            time,
                            sensor: InputSensor::Gyroscope(v),
                        }))
                    }
                    "accelerometer" => {
                        return Ok(Some(SensorData {
                            time,
                            sensor: InputSensor::Accelerometer(v),
                        }))
                    }
                    _ => {
                        warn!("Unknown sensor type: {}", sensor_type);
                        continue;
                    }
                }
            } else if let Some(_frames) = value.get("frames") {
                let input_frame = InputFrame {
                    images: self
                        .video_inputs
                        .iter_mut()
                        .map(|x| x.read())
                        .collect::<Result<Vec<_>>>()?,
                };

                return Ok(Some(SensorData {
                    time,
                    sensor: InputSensor::Frame(input_frame),
                }));
            } else if let Some(groundtruth) = value.get("groundTruth") {
                let pose = load_pose_from_groundtruth(groundtruth); 
                
                return Ok(Some(SensorData {
                    time,
                    sensor: InputSensor::Pose(pose),
                }));
            } else {
                warn!("Unrecognised data format {}", self.line);
                continue;
            }
        }
    }
}
