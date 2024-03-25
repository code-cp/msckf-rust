use anyhow::{anyhow, bail, Context as AnyhowContext, Result};
use clap::Parser;
use ndarray as nd;
use std::path::Path;

use tracing::instrument;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use msckf_rust::config::*;
use msckf_rust::dataset::*;
use msckf_rust::vio::VIO;

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

    // setup logging
    let indicatif_layer = IndicatifLayer::new();
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(indicatif_layer.get_stderr_writer()))
        .with(indicatif_layer)
        .init();

    // load dataset
    let dataset_folder_path = Path::new(&args.input_folder);
    let mut dataset = Dataset::new(&dataset_folder_path)?;

    let header_span = info_span!("header");
    header_span.pb_set_style(&ProgressStyle::default_bar());
    header_span.pb_set_length(dataset.length);

    let header_span_enter = header_span.enter();

    // create vio
    let mut vio = VIO::new();

    loop {
        if let Ok(data) = dataset.next() {
            // process sensor data
            if data.is_none() {
                continue;
            }

            let data = data.unwrap();
            let is_image = vio.process_data(&data)?;
            if is_image {
                Span::current().pb_inc(1);
            }
        }
    }
}
