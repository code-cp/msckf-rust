use anyhow::{anyhow, bail, Context as AnyhowContext, Result};
use clap::Parser;
use msckf_rust::camera::CameraKind;
use msckf_rust::pinhole::PinholeModel;
use std::path::Path;

use indicatif::ProgressStyle;
use tracing::info_span;
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use msckf_rust::camera::Camera;
use msckf_rust::config::*;
use msckf_rust::dataset::*;
use msckf_rust::my_types::*;
use msckf_rust::vio::VIO;
use msckf_rust::kalman_filter::Extrinsics; 

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
    let cameras = load_intrinsics(); 
    let extrisics = load_extrinsics(); 
    let mut vio = VIO::new(cameras, &extrisics)?;

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

fn load_intrinsics() -> Vec<Camera> {
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

    return vec![camera0, camera1]; 
}

/// T_imu_cam: takes a vector from the IMU frame to the cam frame.
/// T_cn_cnm1: takes a vector from the cam0 frame to the cam1 frame.
/// see https://github.com/ethz-asl/kalibr/wiki/yaml-formats
fn load_extrinsics() -> Extrinsics {
        let trans_imu_cam0 = Matrix4d::new(
            0.014865542981794,
            0.999557249008346,
            -0.025774436697440,
            0.065222909535531,
            -0.999880929698575,
            0.014967213324719,
            0.003756188357967,
            -0.020706385492719,
            0.004140296794224,
            0.025715529947966,
            0.999660727177902,
            -0.008054602460030,
            0.0,
            0.0,
            0.0,
            1.000000000000000,
        );

        let trans_cam0_cam1 = Matrix4d::new(
            0.999997256477881,
            0.002312067192424,
            0.000376008102415,
            -0.110073808127187,
            -0.002317135723281,
            0.999898048506644,
            0.014089835846648,
            0.000399121547014,
            -0.000343393120525,
            -0.014090668452714,
            0.999900662637729,
            -0.000853702503357,
            0.0,
            0.0,
            0.0,
            1.000000000000000,
        );

        return Extrinsics {
            trans_imu_cam0,
            trans_cam0_cam1, 
        }; 
}