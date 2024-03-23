use std::path::Path;
use anyhow::{anyhow, bail, Result, Context as AnyhowContext}; 
use ndarray as nd; 

use clap::Parser;

use msckf_rust::dataset::*; 
use msckf_rust::vio::VIO;
use msckf_rust::config::*; 

#[derive(Parser)]
pub struct Args {
  #[clap(short, default_value = "./data/benchmark/euroc/v1-01-easy")]
  pub input_folder: String,
  #[clap(flatten)]
  pub config: Config,
}

fn main() -> Result<()> {
    // parse the config 
    let args = Args::parse();
    let _ = CONFIG.set(args.config);

    // load dataset 
    let dataset_folder_path = Path::new(&args.input_folder);
    let mut dataset = Dataset::new(&dataset_folder_path)?; 

    // create vio 
    let mut vio = VIO::new();

    loop {
        if let Ok(data) = dataset.next() {
            // process sensor data
            if data.is_none() {
                continue; 
            }

            let data = data.unwrap();
            vio.process_data(&data)?; 
        }
    }
}