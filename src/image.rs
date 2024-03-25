/// Row-major grayscale image storage
#[derive(Clone, Debug)]
pub struct Image {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl Image {
    /// Create an empty image
    pub fn empty() -> Image {
        Image {
            data: vec![],
            width: 0,
            height: 0,
        }
    }

    /// Clear the image storage
    pub fn clear(&mut self) {
        self.data.clear();
        self.width = 0;
        self.height = 0;
    }

    /// Get the size for a chosen dimension
    pub fn size(&self, dim: usize) -> usize {
        if dim == 0 {
            self.width
        } else {
            self.height
        }
    }

    /// Linear scaling factor for pixel-space parameters
    pub fn scale(&self) -> f64 {
        (self.width + self.height) as f64 / 1000.
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub fn value(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.width + x]
    }

    #[inline(always)]
    pub fn value_i32(&self, x: i32, y: i32) -> u8 {
        self.data[y as usize * self.width + x as usize]
    }

    #[inline(always)]
    #[cfg(test)]
    pub fn set_value(&mut self, x: usize, y: usize, value: u8) {
        self.data[y * self.width + x] = value;
    }
}
