use image::{Rgb, GenericImageView, Luma};

pub struct FastPoint {
    pub x: u32,
    pub y: u32,
}

mod fast_detector16 {
    use super::*;

    pub const NORTH: usize = 0;
    pub const EAST: usize = 4;
    pub const SOUTH: usize = 8;
    pub const WEST: usize = 12;

    /// The circle with 16 pixels.
    pub const  fn circle() -> [(i32, i32); 16] {
        [
            (0, 3),
            (1, 3),
            (2, 2),
            (3, 1),
            (3, 0),
            (3, -1),
            (2, -2),
            (1, -3),
            (0, -3),
            (-1, -3),
            (-2, -2),
            (-3, -1),
            (-3, 0),
            (-3, 1),
            (-2, 2),
            (-1, 3),
        ]
    }

    pub const fn point(start: u8, index: u8) -> (i32, i32) {
        circle()[(start + index) as usize % circle().len()]
    }

    pub fn detect((x, y): (u32, u32), image: &dyn GenericImageView<Pixel=Luma<u16>>, t: u16, n: u8) -> Option<FastPoint> {
        // exists range n where all entries different than p - t.
        // ignore cardinal direction shortcut.

        let mut consecutive = 0;
        let mut previous = 0;

        let base_v = image.get_pixel(x, y)[0];

        const COUNT: u8 = circle().len() as u8;
        for s in 0..COUNT {
            for i in 0..COUNT {
                let offset = point(s, i);
                let t_x = (x as i32 + offset.0) as u32;
                let t_y = (y as i32 + offset.1) as u32;
                let pixel_v = image.get_pixel(t_x, t_y)[0];

                let delta = pixel_v as i16 - base_v as i16;

                if delta.abs() as u16 >= t {
                    if delta.is_positive() && previous > 0 {
                        consecutive += 1;
                    } else  if delta.is_negative() && previous < 0 {
                        consecutive += 1;
                    } else {
                        consecutive = 0;
                    }
                } else {
                    consecutive = 0;
                }
                previous = delta;

                if consecutive >= n {
                    return Some(FastPoint{x, y});
                }
            }
        }
        None
    }
}

pub struct FastConfig {
    /// Value to be exceeded.
    pub thresshold: u8,
    /// Count of consecutive pixels
    pub count: u8,
}

pub fn detector(img: &dyn GenericImageView<Pixel=Rgb<u8>>, config: &FastConfig) -> Vec<FastPoint> {
    let luma_view = crate::util::Rgb8ToLuma16View::new(img);

    let (width, height) = img.dimensions();

    let mut r = vec![];

    for y in 3..(height - 3) {
        for x in 3..(width - 3) {
            if let Some(p) = fast_detector16::detect((x, y), &luma_view, config.thresshold as u16 * 3, config.count) {
                r.push(p);
            }
        }
    }
r

}


