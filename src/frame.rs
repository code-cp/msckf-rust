use anyhow::Result;

use crate::dataset::*;
use crate::image::Image;
use crate::pyramid::Pyramid;

#[derive(Debug)]
pub struct PyramidFrame {
    /// original image
    pub image: Image,
    /// downsized images
    pub pyramid: Pyramid,
}

impl PyramidFrame {
    pub fn get_image_at_level(&self, level: usize) -> &Image {
        if level == 0 {
            &self.image
        } else {
            &self.pyramid.levels[level-1] 
        }
    }
}

#[derive(Debug)]
pub struct Frame {
    pub pyramid_frames: Vec<PyramidFrame>,
}

impl Frame {
    pub fn new(input_frame: &InputFrame, unused_frame: Option<Frame>) -> Result<Frame> {
        let mut frame = if let Some(mut unused_frame) = unused_frame {
            // Move data buffer from old unused frame to the new frame to avoid allocation
            for i in 0..unused_frame.pyramid_frames.len() {
                unused_frame.pyramid_frames[i].image.clear();
            }
            unused_frame
        } else {
            let mut pyramid_frames = vec![];
            // iterate left and right stereo image
            for image in &input_frame.images {
                pyramid_frames.push(PyramidFrame {
                    image: (*image).clone(),
                    pyramid: Pyramid::empty(),
                });
            }
            Frame {
                pyramid_frames: pyramid_frames,
            }
        };

        let lk_levels = 3;

        for (i, pyramid_frame) in frame.pyramid_frames.iter_mut().enumerate() {
            pyramid_frame
                .image
                .data
                .extend(input_frame.images[i].data.iter());
            pyramid_frame.image.width = input_frame.images[i].width;
            pyramid_frame.image.height = input_frame.images[i].height;
            pyramid_frame
                .pyramid
                .compute(&pyramid_frame.image, lk_levels)?;
        }

        Ok(frame)
    }
}
