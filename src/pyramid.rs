use anyhow::{bail, Result};

use crate::image::Image;

const NAIVE_DOWNSCALE: bool = false;

#[derive(Debug)]
pub struct Pyramid {
    pub levels: Vec<Image>,
    pub parent_size: [usize; 2],
}

impl Pyramid {
    pub fn empty() -> Self {
        Self {
            levels: vec![],
            parent_size: [0; 2],
        }
    }

    pub fn compute(&mut self, frame: &Image, level_count: usize) -> Result<()> {
        self.compute_levels(frame, level_count)?;
        self.parent_size = [frame.width, frame.height];
        Ok(())
    }

    fn compute_levels(
        &mut self,
        frame: &Image,
        level_count: usize,
    ) -> Result<()> {
        while self.levels.len() < level_count {
            self.levels.push(Image::empty())
        }
        if level_count == 0 {
            return Ok(());
        }
        downscale(&frame, &mut self.levels[0])?;
        for i in 0..(level_count - 1) {
            let rest = &mut self.levels[i..];
            // split_first_mut Returns the first and all the rest of the elements of the slice, or None if it is empty
            if let Some((parent, rest)) = rest.split_first_mut() {
                downscale(&parent, &mut rest[0])?;
            }
        }
        Ok(())
    }
}

/// downscale the parent image and store the result in child 
fn downscale(parent: &Image, child: &mut Image) -> Result<()> {
    let w = parent.width as i32;
    let h = parent.height as i32;
    if w % 2 != 0 || h % 2 != 0 {
        bail!("cannot downscale image with shape {w} x {h}");
    }

    let w_half = w / 2;
    let h_half = h / 2;
    child.data.clear();
    child.width = w_half as usize;
    child.height = h_half as usize;

    let v = |mut x: i32, mut y: i32| -> u16 {
        // prevent pixel out of bounds
        if x < 0 {
            x = 0
        }
        if y < 0 {
            y = 0
        }
        if x >= w {
            x = w
        }
        if y >= h {
            y = h
        }
        parent.value(x as usize, y as usize) as u16
    };

    for y in 0..h_half {
        let y2 = 2 * y;
        for x in 0..w_half {
            let x2 = 2 * x;
            let value = if NAIVE_DOWNSCALE {
                (v(x2, y2) + v(x2 + 1, y2) + v(x2, y2 + 1) + v(x2 + 1, y2 + 1)) / 4
            } else {
                v(x2, y2) / 4
                    + (v(x2 + 1, y2) + v(x2 - 1, y2) + v(x2, y2 + 1) + v(x2, y2 - 1)) / 8
                    + (v(x2 + 1, y2 + 1)
                        + v(x2 - 1, y2 - 1)
                        + v(x2 - 1, y2 + 1)
                        + v(x2 + 1, y2 - 1))
                        / 16
            };
            child.data.push(value as u8);
        }
    }
    Ok(())
}
