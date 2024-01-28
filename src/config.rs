use std::{iter::Once, sync::OnceLock};
use clap::Parser;

pub static CONFIG: OnceLock<Config> = OnceLock::new(); 

#[derive(Debug, Default)]
#[derive(clap::Parser)]
pub struct Config {
    #[clap(long, default_value = "0")]
    pub seed: u64, 

    #[clap(long, default_value = "9.81")]
    pub gravity: f64, 

    #[clap(long, default_value = "20")]
    pub window_size: usize, 
}

#[derive(Parser)]
struct Args {
  #[clap(short, default_value = "./data/benchmark/euroc/v1-01-easy")]
  input_folder: String,
  #[clap(flatten)]
  config: Config,
}



