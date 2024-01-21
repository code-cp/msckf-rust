use std::path::Path;
use anyhow::{anyhow, bail, Result, Context as AnyhowContext}; 
use rerun::RecordingStreamBuilder;
use ndarray as nd; 

use vins_rust::dataset::*; 

fn main() -> Result<()> {
    // visualization 
    let rec = RecordingStreamBuilder::new("vins").save("./logs/my_recording.rrd")?;

    // load dataset 
    let dataset_folder_path = Path::new("./data/benchmark/euroc/v1-01-easy");
    let mut dataset = Dataset::new(&dataset_folder_path)?; 

    loop {
        if let Ok(data) = dataset.next() {
            // process sensor data
            if data.is_none() {
                continue; 
            }

            let data = data.unwrap();
            // println!("Got sensor data at time: {:?}", data.time);
            match data.sensor {
                InputSensor::Frame(ref frame) => {
                    // check the image size of stereo images 
                    assert!(frame.images[0].width > 0 && frame.images[0].height > 0);
                    assert!(frame.images[1].width > 0 && frame.images[1].height > 0);

                    let image0_nd = nd::Array::from_shape_vec(
                        (frame.images[0].height, frame.images[0].width), 
                        frame.images[0].clone().data,  
                    ).unwrap();

                    let image0_rerun = rerun::Image::try_from(image0_nd.clone())?;
                    rec.log("world/camera/image", &image0_rerun)?;
                }, 
                _ => {}, 
            }
        }
    }
}