use anyhow::{anyhow, bail, Context as AnyhowContext, Result};
use clap::Parser;
use msckf_rust::camera::CameraKind;
use msckf_rust::pinhole::PinholeModel;
use ndarray as nd;
use std::path::Path;

use indicatif::ProgressStyle;
use tracing::info_span;
use tracing::instrument;
use tracing::Span;
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{filter::LevelFilter, prelude::*};

use msckf_rust::camera::Camera;
use msckf_rust::config::*;
use msckf_rust::dataset::*;
use msckf_rust::my_types::*;
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
        .with(tracing_subscriber::fmt::layer().with_writer(indicatif_layer.get_stdout_writer()))
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
    let camera_kind = CameraKind::Pinhole;
    let camera_matrix0 = Matrix3d::new(458.654, 0., 367.215, 0., 457.296, 248.375, 0., 0., 1.);
    let camera_model0 = Box::new(PinholeModel::new(camera_matrix0));
    let camera0 = Camera {
        kind: camera_kind,
        model: camera_model0,
        image_shape: (752, 480),
    };

    let camera_kind = CameraKind::Pinhole;
    let camera_matrix1 = Matrix3d::new(457.587, 0., 379.999, 0., 456.134, 255.238, 0., 0., 1.);
    let camera_model1 = Box::new(PinholeModel::new(camera_matrix1));
    let camera1 = Camera {
        kind: camera_kind,
        model: camera_model1,
        image_shape: (752, 480),
    };
    let mut vio = VIO::new(vec![camera0, camera1]);

    loop {
        if let Ok(data) = dataset.next() {
            // process sensor data
            if data.is_none() {
                continue;
            }

            let data = data.unwrap();
            vio.process_data(&data)?;
        } else {
            break;
        }
    }

    std::mem::drop(header_span_enter);
    std::mem::drop(header_span);

    Ok(())
}
