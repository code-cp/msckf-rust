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

    #[clap(long, default_value = "1e-2")]
    pub acc_var: f64,

    #[clap(long, default_value = "1e-5")]
    pub gyro_var: f64,

    #[clap(long, default_value = "1e-6")]
    pub bias_gyro_var: f64,

    #[clap(long, default_value = "1e-4")]
    pub bias_acc_var: f64,
}





