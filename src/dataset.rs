use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use anyhow::{anyhow, bail, Result, Context as AnyhowContext}; 
use serde::Deserialize; 
use log::{debug, error, info, warn, LevelFilter};

use crate::image::Image; 
use crate::video::VideoInput; 
use crate::my_types::*; 

pub struct Dataset {
    reader: BufReader<File>, 
    line: String, 
    video_inputs: Vec<VideoInput>, 
}

pub struct InputFrame<'a> {
    pub images: Vec<&'a Image>, 
}

pub enum InputSensor<'a> {
    Gyroscope(Vector3d),
    Accelerometer(Vector3d), 
    Frame(InputFrame<'a>), 
}

pub struct SensorData<'a> {
    pub time: f64, 
    pub sensor: InputSensor<'a>, 
}

impl Dataset {
    pub fn new(path: &Path) -> Result<Dataset> {
        let file: File = File::open(path.join("data.jsonl"))?; 
        let video_inputs = vec![
            VideoInput::new(&path.join("data.mp4"))?,
            VideoInput::new(&path.join("data2.mp4"))?,
          ];
          Ok(Dataset {
            reader: BufReader::new(file),
            line: String::new(),
            video_inputs,
          })
    }

    pub fn next(&mut self) -> Result<Option<SensorData>> {
        loop {
            self.line.clear(); 
            match self.reader.read_line(&mut self.line) {
                Ok(0) => return Ok(None), 
                Err(err) => bail!("Failed to read line {}", err), 
                _ => {}, 
            }

            let value: serde_json::Value = serde_json::from_str(&self.line)
                .context(format!("JSON deserialization failed for line: {}", self.line))?;
            let value = value.as_object().ok_or(anyhow!("JSON line is not a map"))?; 
            
            let time = value["time"].as_f64().ok_or(anyhow!("Time is not a number"))?;

            if let Some(sensor) = value.get("sensor") {
                let v = &sensor["values"].as_array().ok_or(anyhow!("Sensor values not an array"))?; 
                let v: Vec<f64> = v.iter().map(|x| x.as_f64().unwrap()).collect(); 
                assert!(v.len() >= 3, "Sensor values array must contain at least 3 elements");
                let v = Vector3d::new(v[0], v[1], v[2]); 

                let sensor_type = sensor["type"].as_str().ok_or(anyhow!("Sensor type missing"))?;
                match sensor_type {
                    "gyroscope" => return Ok(Some(
                        SensorData {
                            time, 
                            sensor: InputSensor::Gyroscope(v), 
                        }
                    )),
                    "accelerometer" => return Ok(Some(
                        SensorData {
                            time, 
                            sensor: InputSensor::Accelerometer(v),
                        }
                    )),
                    _ => {
                        warn!("Unknown sensor type: {}", sensor_type);
                        continue; 
                    }, 
                }
            } else if let Some(_frames) = value.get("frames") {
                let input_frame = InputFrame {
                    images: self.video_inputs.iter_mut().map(|x| x.read()).collect::<Result<Vec<_>>>()?, 
                }; 

                return Ok(Some(
                    SensorData {
                        time, 
                        sensor: InputSensor::Frame(input_frame),
                    }
                ));
            } else if let Some(_) = value.get("groundTruth") {
                // pass 
            } else {
                warn!("Unrecognised data format {}", self.line);
                continue; 
            }
        }
    }
}



